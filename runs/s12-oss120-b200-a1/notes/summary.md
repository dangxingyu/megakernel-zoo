# gpt-oss-120b decode megakernel on one B200 -- summary (updated 2026-11-18, ~29.5 of 36 agent-h)

Results over HTTP (the score; SGLang 2197 / 3424 us, vLLM 2453 / 3747 at B=1 / B=8):
- Track B (checkpoint formats + a certified int8 sketch of the head): best B=1 s0051 1686.6 us, best B=8 s0056
  2991 us (the current engine = s0056: 1689.8 / 2991) = 1.303x / 1.145x SGLang -- the 1.10x claim holds at both
  batch sizes (milestone h0_step since s0036).
- Track A (checkpoint formats only): best B=1 s0057 1759.7 us (the current track A build = s0057, B=8 3067), best
  B=8 s0050 3061 us = 1.249x / 1.119x SGLang -- both tracks clear 1.10x at both batch sizes.
58 submissions, 56 valid (s0002 build failure and s0003 error, both early); every one since s0004 passed all gates.
Alone == batch is bitwise in track A; in track B since s0054 the head logits differ in the last bits between the
two kernel instances (max 5.5e-5), tokens identical. B=8 also depends on the prompt set by up to ~5 %
(findings.md), which bounds how finely the B=8 score can be tuned.
Build: `engine/TRACK` = A or B (B adds `-DHSK=1`); several defaults follow the track (SPF, PF_STAGES_W, ETAIL,
HDYN / HTAIL, HSK_RSK, CT_ORIG, ...): the two builds allocate registers differently and want different values.

## Engine (engine/mk.cu, engine/api.py)
- One cooperative persistent kernel (152 blocks x 256 threads, one per SM) runs prompt + greedy steps (serve mode:
  tokens to host-mapped memory, polled by `poll()`); two instances: B <= 2 ("flat") and B >= 3 ("wide").
- Per layer: A norm1 + qkv (RoPE, K/V cache writes in mma fragment order; track B: L2 prefetch of the K/V chunks and
  small parameters later phases read; track B, B <= 2: 2 qkv rowtiles per block, the 16 extra V rowtiles as
  segment tasks spread over 40 blocks, V(pos) summed by the attention warp that needs it), B attention (64-key chunks; units of <= 8 (flat) / 16 (wide) chunks on one
  block with a shared-memory merge, longer ones through global partials), C o_proj + residual, D norm2 + router
  (32 blocks, K in quarters), E top-4 + union + gate_up (MXFP4 via scaled e2m1 cvt), F down (per-slot completion
  counters instead of a barrier), wide: a combine step.
- Weights stream through per-warp TMA rings (3 stages x 9 items) with task tables; the next phase's first stages are
  prefetched before each grid barrier (2 stages; 3 in track A's wide instance). Tasks are cut on whole stages.
- Track B head: certified int8 sketch of lm_head (605 MB instead of 1.16 GB) with per-row error-block norms; the
  rows whose upper bound reaches the best lower bound (median 2 per position) are recomputed with the exact head's
  own arithmetic; tokens are exactly the BF16 head's argmax (generate == step-greedy, http_tokens 8/8).

## What mattered (journal.md has every measurement)
1. Per-warp TMA rings with cross-barrier prefetch, fragment-order operands, scaled fp4 cvt: 3613 -> ~2000 us (B=1).
2. Removing serial dependencies in the latency-bound phases: attention split + merge, E->F counters, batched loads,
   bulk-copy staging.
3. Task granularity: per-warp stage chains, not per-SM bytes, bound the dense phases; whole-stage tasks.
4. The head sketch (track B): -60..-75 us at B=1, ~-40 us at B=8 (wide sketch out of line; candidate rescoring with
   pooled ring slots; whether candidate blocks copy or re-norm the head input flipped with the allocation: copy
   -15 at s0034, re-norm -24 at s0045).
5. K/V cache (and small per-layer parameters) prefetched into L2 at the start of phase A (they do not survive a
   step in L2): B=1 -20, B=8 -21 us (track B). B <= 2 qkv on 2 rowtiles per block (QXO): B=1 -4.5 on one pod (s0047), +1.3 on the next: pod-dependent.
   Track A's last steps (s0048-s0050): the exact head's last rowtiles split into K halves over the warps (HTAIL),
   then the K/V prefetch in the B >= 3 instance, which only paid after HTAIL had moved the allocation: B=8 3114 -> 3061.
6. Measuring B=8 on the scored workload: the HTTP client's word-salad prompts route far more alike than real text
   (3142 vs 3644 us in process); re-tuning same-bits switches on a bench that reproduces them (bench.py --synth)
   found -60 us at B=8 (s0034-s0038) that the real-text bench did not show, and re-sweeps after every larger change
   another -40 (s0044-s0046: switch values flip with the register allocation).
Dead ends (all measured): L2 prefetch of weights (incl. predicted experts), counters or cluster barriers instead of
grid barriers, deeper rings, split o_proj rowtiles across blocks (qkv: kept only in the B <= 2 instance), last-arriver
combine, adaptive head bounds,
many register-pressure rewrites (both instances sit at 255 registers: even unexecuted code moves performance by
10-50 us, and switch values flip sign as the allocation moves).
Method notes: judge bit-changing variants on fixed content (bench.py --fixed); never let a failed compile reuse an
old library (tools/abs.sh deletes both first -- one KVPF null result came from that); keep both instances' sums in
the same order (greedy alone vs batched diverges within tens of tokens otherwise).

Reproducibility (checked at ~23.5 agent-h): HEAD with TRACK=A is byte-identical to s0050's submitted source; HEAD with
TRACK=B differs from s0047's submitted source only in track-A-only code and defaults (HTAIL / HTS, per-instance KVPF)
and builds the same kernel in effect: logits bitwise equal to position 1100 at B=1 and B=8, same in-process speed.

## Reproduce and measure (tools/)
- Build: `echo B > engine/TRACK` (or A); `bash engine/build.sh`. Full pre-submission check (compiles both tracks,
  long-context, reference, bench, edge cases): `bash tools/presubmit.sh`.
- HTTP-like B=8 timing: `python tools/bench.py --b 1,8 --gen 1024 --synth 1 [--seed S]` (the client's kind of prompts);
  fixed content for bit-changing variants: `python tools/bench.py --b 1,8 --fixed tools/study/synth_fixed.pt`.
- Same-card A/B of a source variant: `ABS_EXTRA=-DHSK=1 ABS_BENCH="--synth 1" bash tools/abs.sh <dir with mk.cu> 3 ...`
  (`ABS_EXTRA_A` adds flags to the variant only); bitwise comparison: `python tools/cmp_long.py libA.so libB.so`.
- Phase profile: `MK_VARIANTS="prof:-DMKPROF" bash engine/build.sh; python tools/prof_run.py --b 8 --pos 600 --synth 1`.
- Track B token check: `python tools/hsk_check.py`, `python tools/study/sketch_verify.py <seed> <B>`.

## Final summary
Best submissions (HTTP, us per step; SGLang 2197 / 3424, vLLM 2453 / 3747):
- Track B (checkpoint formats + certified int8 head sketch): **s0051 B=1 1686.6** (1.303x SGLang), **s0056 B=8 2991**
  (1.145x). engine/ (TRACK=B) builds s0056's kernel exactly (SASS-identical; HTTP 1689.8 / 2991); s0051 is the
  same B <= 2 code as s0056 minus the barrier back-off (its B=1 is within the +-4 us evaluation noise of s0056's).
- Track A (checkpoint formats only, `echo A > engine/TRACK`): **s0057 B=1 1759.7** (1.249x; the track A build
  is s0057's kernel, SASS-identical; B=8 3067), best B=8 **s0050 3061** (1.119x).
Both tracks clear the project's 1.10x line at both batch sizes; all gates passed.

What worked (largest first): per-warp TMA rings with cross-barrier prefetch and fragment-order operands; removing
serial dependencies in the latency-bound phases (attention split/merge, E->F counters, bulk-copy staging);
whole-stage task cuts (per-warp stage chains, not bytes, set the dense phases); the certified int8 head sketch
(track B); prefetching the K/V cache and small per-layer parameters into L2 at the start of each layer; measuring
B=8 on prompts like the HTTP client's (bench.py --synth) and re-sweeping same-bits switches after every larger
change, because the register allocation (both instances at 255 registers) flips which switch values win; in track A,
splitting the exact head's last rowtiles over the warps (HTAIL).

What did not: L2 prefetch of weight streams (incl. predicted experts), deeper rings, counters/cluster barriers
instead of grid barriers, last-arriver combines, cross-block splits of o_proj, adaptive head bounds, staged norms,
compiler options and layout offsets; many structural changes that showed their intended effect in the profile
(e.g. qkv rebalancing: phase A -0.9 us/layer) but lost it to register allocation in the wide instance.

What I would try next:
1. Make structural gains survive the normal build. In this run's last third, the profiling build (-DMKPROF)
   repeatedly showed the intended effect of a change that the normal build then lost to register allocation:
   K-split candidate rescoring in the B <= 2 instance (head -3.5 us), RMSNorm sums spread over the 8 warps
   (NORM_G8: -12 us/step), phase D's norm input staged in shared memory (-0.23 us/layer). The marks are not acting as
   scheduling barriers (MARK_SB did not reproduce it). Candidates: each phase body as its own out-of-line function
   with little live state across the call (attention alone out of line was +40..+107: the call has to be cheap),
   or the two instances as two separate kernels compiled with their own limits.
2. Then: the summation-order changes above (NORM_G8 form 2 is cheap in the B >= 3 instance), the B <= 2 rescoring
   split, and the 1.5-2 us start latency of phase E (experts are known only after the router).
3. Long-context attention: the B=1 attention phase grows from 4.5 us/layer at position 100 to 7.0 at 1000
   (~55 us of the step over the HTTP decode window); above 8 chunks per KV head it goes through global partials.
4. Judge balancing changes on the scored card: per-SM bandwidth differs between cards, so a same-card A/B on
   cards 1-3 does not predict card 0 for them (SDYN); and always check that an A/B base is SASS-identical to the
   last submission when code shared by both tracks has changed.

## Extension (24 -> 36 agent-h), so far
- s0052 (track B) / s0053 (track A): the grid barrier arrives with one red.release add and polls the counter with
  a 40 ns back-off (microbenchmark 1.345 -> 1.155 us per barrier; 30 / 60 ns the same). In the engine the gain
  depends on the card (B=8 -5..-14 us on three cards; B=1 -7, -6, +1); over HTTP: track B 1689.8 / 3019, track A
  1770.9 / 3075.
- s0054 (track B, B=8 3019 -> 3007): candidate head tiles rescored by all 8 warps as 4 K segments each instead of one
  warp per tile (~8.5 us consume chain). The B >= 3 instance's exact head now sums every rowtile as 4 K segments in
  order, so the rescoring stays its exact argmax; the B <= 2 instance is untouched (the same split there saved 3.5 us
  of head time in the profile but cost ~8 us of allocation). Price: head logits alone vs batched differ in the last
  bits (max 5.5e-5); greedy tokens alone vs batched still identical (8 x 1024).
- s0055 (track B, B=8 3000): re-sweep after s0054: HSK_XG 2 (the published normed head input by bulk copy) -8.
- s0056 (track B, B=8 2991): the wide sketch prep takes its digit scale from an upper bound of the normed maximum
  (max |w x| ri (1 + 2^-6)), so norm and digits are one pass. A fully unrolled first form was +40 us (the prep runs
  once per step from a cold instruction cache); with rolled loops -6.
- s0057 (track A, B=1 1770.9 -> 1759.7, B=8 3067): the exact head's last ~8 % of rowtiles claimed one at a time from
  a global counter by whichever warp is free (HDYN); with it the B <= 2 K/V prefetch and qkv split win again in
  track A. The head's blocks already stream at ~7.3 TB/s together, so balancing itself is worth ~4 us; the rest is
  allocation.
- s0058 (track B, failed: 1692 / 3010): the same dynamic tail for the sketch (SDYN) looked like -6 / -7 on cards 1-3,
  but HDYN had restructured the shared task consumer for every list; the A/Bs compared SDYN on top of that. With
  the old form restored (CT_ORIG) SDYN is +7..+10 / +38..+46 and the restructure alone +8..+16 / +25..+53. Both
  tracks' builds are again SASS-identical to their last submissions (s0056, s0057).
- Measured and dropped: attention as an out-of-line function (+40..+107 us: a call in the layer loop saves every
  live register), expert speculation from the previous step (reuse only 28 % at B=1), sketch tail split, per-phase
  prefetch depths, early rebuild of the head input, back-off in other polls, arrive-before-prefetch, a router
  counter instead of the D -> E barrier, register caps, layout offsets.
- Checks: warp-stall sampling (block barriers 45 % of stall cycles at B=1, instruction fetch 4 %); the B=8 choices
  hold on other prompt sets.
