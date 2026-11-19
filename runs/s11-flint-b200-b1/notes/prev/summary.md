# Summary of the run (Flint on one B200: 24 agent-hours, then an extension of 12 for batch 8 on track A)

## The best submission and its number (state after the extension)

`engine/` as committed is track A. All figures: sealed evaluations, HTTP, us per token at batch 1 / us per step at
batch 8; vLLM 0.33.0 is 1971 / 3796 in the trial and 1990 / 3810 in the operator's same-card rematch.

| | submission | batch 1 | batch 8 | against vLLM (trial) |
|---|---|---|---|---|
| track A, end of the extension | s0077 (= `engine/`: 1479 / 3397); best figures of the same engine family: s0075 at batch 1, s0076 at batch 8 | 1478-1483 | 3394-3397 | 1.33x / 1.117-1.118x |
| track A, three hours before the end | s0070, and again as s0071 | 1493 | 3426 | 1.32x / 1.108x |
| track A, end of the first 24 hours | s0058 | 1520 | 3877 | 1.30x / 0.98x |
| float32 residual stream (was called track B; the audit showed it is not bit-identical to track A) | s0057 | 1475 | 3950 | 1.34x / 0.96x |

The extension's goal was 3470 us per step at batch 8 on track A (1.10x of the same-card rematch, 3810) without
giving up batch 1. Reached: 3394 (s0076; 3397 in s0075, 3426 in s0070), with batch 1 at 1478-1484 (about 40 us
better than before). That is 76 us inside the goal; the spread of one evaluation is about 30.
77 submissions, 76 valid; the nineteen of the extension (s0059-s0077) are all track A and all have the bits of s0058
(tools/digest_long.py, digest1.py unchanged; bcheck, flake, api_stress, long_ctx and the watchdog test before each).
On the host-driven path without HTTP (the evaluator's `w1_like`: one launch per step, the chosen token read on the
host) s0076 is 1512 / 3579 us (batch 1 was 1512 to 1579 in the last five evaluations, whose batch-1 kernels differ
by 18 us): 1.25x to 1.30x at batch 1 and 1.06x at batch 8; a launch, the host's argmax and the round trip
cost 80-170 us per step there, which the headline path (generate / serving, one launch per cohort) does not pay.
The two defects of the audit are fixed in s0059: a served request that does not fit the context and a batch over
max_batch are refused with an exception.

## What worked in the extension (findings 15a-15g, journal under "Extension")

| change | batch 1 | batch 8 |
|---|---|---|
| s0060: the mma operand getter without a conditional load (the compiler predicated it or left a branch, 5 to 23 instructions per tile, depending on unrelated code) | 1531 | 3898 -> 3842 |
| s0061: hand-over counters polled from warp 6, not from the leader warp 0 (a warp's load waits for the exchanges it issued before) | 1531 -> 1514 | 3829 |
| s0062: the attention's K/V tile batches as volatile asm (the compiler had moved plain loads down to their uses), and used in the batch > 2 kernel | 1514 -> 1494 | 3829 -> 3784 |
| s0063: copy of the attention output with 4 loads in flight | 1497 | 3784 -> 3735 |
| s0064: down phase with one fragment copy up front (8 slots per sequence), rounds stream tiles only | 1496 | 3735 -> 3577 |
| s0066, s0067: two rewrites that took values off the stack (a pointer array in the new copy; lane-indexed arrays in the routing merge) | 1493 | 3571 -> 3527 -> 3499 |
| s0068: batch 7-8, the output projection of the 8 main blocks with an attention pair computed by 8 other main blocks | 1493 | 3499 -> 3460 |
| s0069: those hosts on blocks 24..31 instead of 8..15 | 1493 | 3460 -> 3448 |
| s0070: the copy of the attention output asks for all its pending words in each attempt | 1493 | 3448 -> 3426 |
| s0072: the routing's union with two barriers fewer (a barrier of a 16-warp block is about 0.1 us) | 1491 | 3426 -> 3412 |
| s0073: the routing's candidates (batch > 2) by two rank counts through shared memory instead of six rounds of shuffles | 1490 | 3412 -> 3405 |
| s0074: the routing's merge at batch 1-2 over 27 pruned candidates, one per lane, instead of 48 | 1490 -> 1484 (in process 1382 -> 1364) | 3408 |
| s0075: the routing union's last pass on all 16 warps (4 sequences per thread instead of 8 on half the warps) | 1478 | 3408 -> 3397 |
| s0076: the routing merge ends at the union's table bytes; sum and weights on the idle warps of the next stage | 1483 | 3397 -> 3394 |
| s0077: the union's last pass without its loop over the per-warp counts (-4 us in process on one card, 0 on another; not visible in the sealed result) | 1479 | 3394 -> 3397 |

How they were found matters more than the list. The first two came out of reading the disassembly and the profile
for things that did not add up; the last three out of two methods adopted half-way: decide on the scored workload
(prompt 64, 1024 greedy tokens: positions 64..1088, tools/gen_long.py and serve_long.py), and ask "what would the
step be without X" of the normal build (`-DWHATIF`), not of the profiling build. The last six (s0072-s0077) came
in the final three hours from the third method, the cycle counter (`-DMK_CLK`, tools/bt_clk.py: stamps that touch
no memory), pointed inside the routing: its top-6 selection was 3.5 us of a 70 us layer at batch 8 and 2.6 of 28 at
batch 1, in chains of shuffles and in one warp ranking 48 candidates; findings 15m. Had this tool existed at hour
24, the routing would have been the first thing it showed.

## What did not work in the extension

- Taking work off the 8 main blocks that also carry an attention pair, as long as the down phase was the place
  where every block waited: AFIRST (their q/k/v projection hosted by other blocks, +310 us), a weighted split of
  the expert phase, sentinels or tiles moved in front of their attention; HOSTC (their output projection hosted)
  was +10 us then and -63 after the down phase was rebuilt (s0068).
- Orders that put tiles in front of exchanges or copies behind tiles (FPRE_EARLY, F_PIPE), plain stores for values,
  a counter instead of sentinels for the hidden units, two V batches in flight, batched norm loads (6th attempt).
- A counter instead of sentinels for the expert phase's hidden units (E_CTR): 17 us faster and not strict (4
  mismatches in 398 batch-8 repetitions of the repeat test; digests and bcheck had passed). Off.
- The same rewrites on the batch 1-2 kernel: its allocation (spill line 176/20/16) is a good one, and three changes
  that left batch 8 alone cost batch 1 30-64 us.
- A speculative prefetch of expert weights was measured, not built: 36% of a step's units repeat from the previous
  position, the L2 keeps about 64 MB and loses them to 48 MB of other traffic (findings, journal).

## Where the batch-8 step is now, and what I would try next

A layer at batch 8 on the scored kind of text takes about 70 us (tools/bt_clk.py: stamps from the cycle counter in
the normal kernel, findings 15l): 40 us in the two streamed phases, which are within 5 us of their memory time, and
30 us in front of them: six hand-overs through memory at 2-2.5 us each from the writer's exchange to the reader's
value, the attention (4-8 us after q/k/v arrive, growing with the context), the routing (3.7), the norms.

1. Fewer or cheaper hand-overs. An exchange takes about 2 us to be seen by another SM and so does a plain store;
   a counter bumped after the data is not a hand-over (E_CTR: faster and not strict). What is untried: the release /
   acquire forms of the loads and stores that finding C11 of the Qwen3.8-27B run describes, and merging phases so that
   a layer needs five hand-overs instead of six.
2. The routing computation in every main block (about 200 us per step by timing study): rank raw logits with an
   exact fallback for ties of the rounded sigmoid, fewer barriers in the union.
3. The hosts' second pass (their own output projection comes 1.9 us after everybody else's): needs 32 tiles per warp
   at hand when the attention output arrives; every arrangement I thought through moves the same trip to memory
   somewhere else.
4. The batch 1-2 kernel's register allocation: it is 35% slower in a bad one (1481 -> 2005 us with study code
   compiled in), so a better one than 176/20/16 bytes of spills may exist; nothing was searched there in the
   extension.
5. The host-driven step path at batch 8 (1.07x on the evaluator's `w1_like`): one launch per step and the host's
   argmax cost 80-170 us per step; a kernel that stays alive across host steps cannot coexist with the host's own
   kernels on this card.

---

# The first 24 hours (as written on 2026-11-12)

### Addendum of the last three hours (s0072-s0077; findings 15m)

Best submission: `engine/` = s0077, track A, bits of s0058: 1479 us per token at batch 1 and 3397 us per step at
batch 8 over HTTP (1.33x / 1.117x of vLLM's 1971 / 3796). s0075 (1478 / 3397) and s0076 (1483 / 3394) are the same
engine to the evaluation's precision; in process s0077 is the fastest of the three (generate 3400 / 1363).

What worked: cycle-counter stamps INSIDE the routing's top-6 selection (3.5 us of a 70 us layer at batch 8, 2.6 of
28 at batch 1), then six rewritings with the same order and ties: two barriers fewer in the union; rank counts
through shared memory instead of six rounds of shuffles; the batch-1 merge over 27 pruned candidates instead of 48;
the union's last pass on all 16 warps; the weights off the merge's path; the per-warp counts without a loop.

What did not work there: one warp per sequence (RT8, +13 us); sorted rounds (RSORT=1, -2.5: the rounds are bound by
their shuffles); tie terms by a warp match (RMATCH: nothing at batch 8, +30 at batch 1, and one part hung through my
error); batch-1 weights from shared memory (RTSW, +54) and branch-free candidate stores (TRIBF, +18), both through
the register allocation; one branch for the four sigmoid divisions (SIG4, -1.5, not taken). Predictions from
instruction counts were off by a factor of two to five, and the last two submissions are below what one sealed
evaluation resolves.

What I would try next: (1) the same stamps inside the next routines on the batch-1 path, where nothing has been
looked at this way: the 2.8 us between routing and first expert tiles, and the attention's scores stage; (2) at
batch 8 the candidate stage (1.5 us per layer) from its disassembly (tools/sass_route.py), not from another
rewriting: ranking raw logits is exact (the compiled sigmoid is monotone over all floats, tools/sig_mono.cu) but
RMATCH says fewer instructions alone do not pay there; (3) the four barriers of the batch > 2 routing; (4) the
older list below (hand-overs at 2-2.5 us each, the norms' sequential loads), which is still where most of the 30 us
in front of the streamed phases are.

## The best submission and its number

`engine/` as committed is the source of the last two submissions (kernel copy `tools/mk_s57.cu.bak`):

| | submission | batch 1, us per token (HTTP) | batch 8, us per step (HTTP) | against vLLM 0.33.0 (1971 / 3796) |
|---|---|---|---|---|
| track B (float32 residual stream, hi+lo router operands) | s0057 | 1475 | 3950 | 1.34x / 0.96x |
| track A (formats and data flow of the general engines) | s0058 | 1520 | 3877 | 1.30x / 0.98x |

The board's best figures, 1473 (s0055, track B) and 1516 (s0056, track A), are the same kernel before the last
change (strong loads for the hand-over reads, no effect on speed); the differences are evaluation noise.
The claim of 1.10x holds at batch 1 on both tracks and not at batch 8 (about 2% behind the general engine).
58 submissions, 57 valid. Every result is strict (0 mismatches in 299 repetitions per track and batch size) and
bitwise the same for a sequence alone and in any batch.

## What worked (in the order found; numbers in notes/findings.md and notes/journal.md)

- One persistent kernel, 152 blocks: 128 stream weights, 24 do attention and the router; shared-memory reservation
  cut to 128 KB; two kernel instances (batch 1-2, batch 3-8).
- Strict hand-overs without a wait at the end of a phase: sentinel words (findings 12d), now read with strong loads
  (12e).
- Routing by ranks, epilogues off the critical warps, router tiles preloaded (8, 8a, 8c).
- K/V cache tiles requested in batches at long contexts (8b).
- The greedy token gathered in one trip (batch 1-2) or 16 strides (batch 3-8) instead of 302 loads in a row.
- Softmax weights by one reciprocal and an exact FMA correction instead of 4224 divisions per pair (13c): the
  largest single gain of the last third, same bits.
- Sentinel retries without the abort word in front of every reload; the hand-over counters and three small buffers
  placed on the fastest of 16 pages by a timing at start-up (13g).
- Track B: the kernel body as an inlined function, which gives the batch-1 kernel the better register allocation
  (13i): 1557 -> 1473.
- Small ones at batch 8: early counter poll on the main blocks with an attention pair (13d), norm scales once per
  warp (13e).

## What did not work

- Speculative, early or certified routing (item 14 and journal): the router blocks are the critical path anyway.
- Other organisations of the down phase, block-major layouts, staging more K/V in shared memory, batched norm
  loads, swapped mma operands, split accumulators, a conflict-free fragment copy (13b, 13e, 13f).
- The split attention pair for batch 7-8 (scores on the main block, V mix on an attention block): correct and
  bit-identical, but slower in every form tried (inlined, as a call, as two kernels), because the kernel has no
  register to spare and two kernels side by side do not run like one (13d, 13h).
- Any other assignment of roles to physical blocks, weighted splits of the expert phase, more placed buffers, 64
  candidate pages, never-taken guards as allocation draws, a 32-bit container for track A's stream (13g, 13i, 13j).

## What I would try next

1. Batch 7-8: give the two roles separate register allocations inside one launch, so that the split attention pair
   (about 150 us per step, the distance to the general engine) can be used.
2. Free a few registers in the hot loops (the staging array takes 64 of 128): it would end the "every change is a
   new allocation draw" problem and may give a better draw than 36/16 bytes of spills.
3. Each hand-over counter on a page of its own chosen at start-up; fewer barriers in the attention.
4. Why main blocks 112..127 are always last at a barrier (13j), and what that says about the real peak rate.
5. Track B: one container per kernel instance (float32 stream helps at batch 1, costs at batch 8).

The full ranked list with the measurements behind it is at the end of notes/findings.md.
