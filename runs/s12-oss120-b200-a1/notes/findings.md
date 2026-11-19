# What is specific to gpt-oss-120b on a B200 (each item with its measurement)

## Card (job card 1, e0001, e0005-e0008)
- 152 SMs, L2 129 MB, 232 KB smem opt-in per block, 12 CPU cores per job.
- bench/stream.cu (152 blocks x 512 threads, 4 GB cold buffer):
  - BF16 mma-fragment tiles (16 B per lane per mma): 7.44-7.47 TB/s (8-16 loads in flight per thread).
  - MXFP4 records 512 B codes + 32 B scales (544 B, not line aligned), raw: 6.65 TB/s.
  - MXFP4 decode by `cvt.rn.bf16x2.e2m1x2` (sm_100a only: F2FP.BF16.E2M1.UNPACK_B, byte select free when the
    word is unpacked with mov.b32 {b0,b1,b2,b3}) + HMUL2 by the scale: 5.7 TB/s; extracting bytes by shift/mask
    instead: 4.6 TB/s. Same with 4 chunks per line-aligned super-record: 6.15 TB/s.
  - cvt + fp32 post-scaling per 32-group (two mmas into a zero accumulator, 4 FFMA by 2^(S-127)): 6.82 TB/s.
    -> the scale multiply is the costly part of the decode, not the e2m1 conversion.
- `cvt.rn.f16x2.e2m1x2` / `.bf16x2.e2m1x2` exist only with -gencode arch=compute_100a,code=sm_100a.

## Checkpoint (e0002, tools/study/scales.py, exact counts over all 36 layers)
- E8M0 scales span 111..138 (2^-16 .. 2^11); 18.7% of groups have S >= 128, so a single bf16 multiply by
  2^(S-1) (the subnormal-placement trick of Marlin) would overflow: the decode must normalise first or scale in fp32.
- FP4 code histogram (low/high nibbles together): +0 7.20e9, +0.5 9.45e9, +1 11.5e9, +1.5 6.43e9, +2 9.41e9,
  +3 6.82e9, +4 4.81e9, +6 1.68e9; negatives the same within 0.1%. -0 (code 8) occurs 7.20e9 times (as often as +0).

## Head and embedding rows (e0208, tools/study/head_rows.py, all 201088 rows, exact)
- No two rows of lm_head (or of the embedding) are bit-identical; no zero rows; no row with < 100 non-zeros.
- Row norms of lm_head: min 0.0149, median 0.265, max 0.439. The last 1100 rows (reserved tokens) have norm ~0.0149
  (embedding ~2.4 vs median 94): they can never win an argmax against a normal logit, but they are 0.55% of the head:
  nothing to gain from skipping them.
- MXFP4 code entropy (order 0, from the e0002 histogram): 3.87 bits per code (vs 4): at most 3% from an entropy code;
  E8M0 scales take 28 distinct values (111..138): a 5-bit code would save 2.2% of the expert bytes. Neither pays the
  decode in a step that is not bound by bytes.

## Expert choice is predictable from the layer input (e0248, tools/study/predict.py)
Own fp32 reference, 8 real texts x 192 positions x 36 layers (55296 token-layers). Predictor: layer L's router applied
to the layer's input x_in (before attention) normed with L's post-attention norm weights; actual: the router on x
after attention. Recall of the actual top-4 within predicted top-k: k=4 0.856, k=5 0.916, k=6 0.945, k=8 0.969,
k=12 0.983. Batch-8 union coverage (share of the step's actual experts inside the predicted union): k=4 0.888
(predicted union 24.9 vs actual 24.5 experts), k=6 0.963 (34.8). L2 of the card: 129 MB.
Allows: pulling the predicted experts' weights into L2 while A-D run (latency-bound, HBM mostly idle) without any
change of arithmetic (track A: only where bytes come from). Does not allow skipping anything: the prediction only
steers prefetches.

## The output head admits a certified int8 sketch (tools/study/hsk_bound.py, e0259 / e0313)
lm_head rows quantized per row to int8 (s = fp32(max|w| / 127)), error-block norms over 64-column blocks rounded up
to BF16; head input as two int8 digits (14-bit fixed point). On 256 real positions (own fp32 reference, BF16 head
input) the certified bound |w.x - approx| <= sum_blk |e_blk| |x_blk| + |s q| |r| (+2^-11 |w| |x| for the exact
head's own fp32 error) holds with max |err| / bound = 0.11; rows that can still be the argmax: median 2 (max 14),
i.e. 2 of 12568 16-row tiles. 5-bit (QMAX 15): median 552 tiles; 6-bit (QMAX 31): 24; int4: all. Allows: a greedy
head that reads 605 MB instead of 1.16 GB and returns exactly the BF16 head's argmax (track B, s0018..s0021).
Does not allow: logits (step()) from the sketch; anything below 8 bits with this bound.

## Runner / memory-system facts measured on this card (bench/*.cu)
Grid barrier floor ~1.2-1.3 us (atomic or red.release + acquire poll); hardware cluster barriers make a two-level
barrier slower (1.5-1.8 us). TMA ring stream from L2 without per-item work 67-90 GB/s/SM, with the fp4 decode
~38 GB/s/SM, from HBM ~41-43 either way. Evict-first bulk reads are not retained in L2. One bulk copy per staged row
(attention output, activation rows) beats per-thread loads + fragment-order smem stores by 0.5-1.3 us per phase,
unless the SM's bulk-copy queue already holds prefetched weight stages (staged norm input: much slower).

## Task-granularity facts of this kernel's runner (journal ~9-11 agent-h, s0023-s0028)
- Dense phases are bound by the longest per-warp chain of ring stages, not by per-SM bytes: o_proj blocks with two
  rowtiles took as long as one-rowtile blocks once every warp held one task (NSEG_O 4), and a bit-identical partner
  split of the second rowtile (CSPLIT) only added cost.
- Every task that ends in a partial ring stage costs a stage turn: o_proj 16 -> 4 segments with bounds on whole 9-item
  stages (63/63/63/67) and the router in 5 one-stage tasks gave -35 us at B=1 and -190 us at B=8 (s0023, s0024).
- More, shorter tasks for per-warp balance only pay when added sparingly: splitting all down units in halves (FHALF)
  cost +163 us at B=8, splitting only the n mod 8 remainder units into one-stage segments (ETAIL/FTAIL) gained.
- Ring depth (RS 4) and prefetch depth (PF_STAGES 1/3) do not help; the fp4 decode caps the L2-hot stream at
  ~38 GB/s/SM.
- Both kernel instances sit at the 255-register ceiling: code that is never executed in an instance still moves
  that instance's allocation (+10..+50 us/step observed); instance-specific code must be behind if constexpr.

## The scored workload's routing at B=8 (e0435-e0439, tools/bench.py --synth 1, tools/study/synth_gen.py)
The HTTP client's prompts are random common words + "Continue the text above with a very long and detailed
passage" (a different seed per stream). With prompts built the same way, in-process B=8 generate/serve takes
3142-3144 us vs 3644 us with the bench's real-text prompts; HTTP s0034 measured 3125 (s0035: bench -10 -> HTTP -10,
s0036: bench -11 -> HTTP -10). The 8 continuations differ (8 distinct streams), but their routing overlaps much more
than real text's. Allows: tuning B=8 for the score on --synth. Does not allow: conclusions about real-text B=8 (the
evaluator's in-process B=8 generate, 3343 us at s0034, sits between the two).
Measurement rules learned: (1) a change that alters any bits also alters free-running text and hence routing; judge
it on fixed content (bench.py --fixed: step() teacher-forced over a saved synthetic continuation) -- RTASK 3 looked
-62 us on free-running serve and was -3 on fixed content; (2) two benchmark jobs at once on cards 2 and 3 slow each
other by ~20 us at B=8.

## Knob values move with the register allocation (s0035, s0036)
Both kernel instances sit at 255 registers; per-instance knobs measured as wins earlier flipped sign later:
FTAIL (s0027: -5 us) was +10 us after s0034; ATT_BLKW 2 (s0027: +15 us) was -11 us after s0035; STEPARG (s0030: -7)
was +7..+11 by then (left on: it does not add with ATT_BLKW 2). Re-sweep same-bits knobs after larger changes.

## Why batch == single bitwise matters here (tools/study/alone_vs_batch.py, ~16 agent-h)
With identical arithmetic in both kernel instances, greedy continuations of 8 real-text prompts are the same alone
and in a batch of 8 for all 1024 tokens. Changing only the B <= 2 instance's qkv K split (10 -> 5 segments: a
different fp32 summation order) makes them diverge after 1..956 tokens (mean matching prefix 0.20). Greedy decoding
amplifies last-bit differences within tens of tokens; any per-instance change of summation order puts the
http_tokens gate (server tokens of the batched check prompts vs the engine's own greedy tokens) at risk.

## The K/V cache is not L2-resident across steps (~16.3 agent-h)
Pulling each block's attention K/V chunks into L2 at the start of phase A (cp.async.bulk.prefetch.L2, one or two
bulk requests per block) speeds up phase B: track B B=1 -14 us, B=8 -21 us per step (HTTP-like bench, bitwise
equal). The weight streams (~4.5 GB per step, evict-first) and the ~360 MB of K/V at B=8 do not leave the cache
lines of the previous step's attention in L2. (Track A's build: +17 us at B=8 -- allocation, not the prefetch.)

## B=8 speed depends on the prompt set (~20 agent-h, tools/bench.py --synth 1 --seed S, engine of s0047)
Six sets of 8 synthetic prompts of the client's kind (seeds 1000..6000): in-process serve B=8 3043, 3135, 3054, 3099,
3209, 3137 us (same build, same card). The routing overlap of the 8 continuations, and so the expert bytes per step,
differs by prompt set: up to +5.5 %. The HTTP client draws new prompts for each of its 3 repeats, which is why one
repeat is often ~7 % slower (s0030-s0047: max/median 1.07-1.08); the median of 3 is still stable to a few us
between evaluations of one engine. Tuning was done on seed 1000 (3043 in process vs HTTP 3017-3021).
Track A's final engine (s0050) on the same six sets: 3080, 3174, 3086, 3139, 3248, 3176 us (up to +5.5 % as well;
HTTP 3061). The ordering of the prompt sets is the same in both tracks: the variation is the routing of the text.

## The final track B engine's tokens on the scored kind of workload (tools/study/sketch_verify.py, ~21.5 agent-h)
generate() (head sketch + exact rescoring of candidates) vs the argmax of step()'s exact BF16 logits, teacher-forced
over the same 1024 generated tokens, synthetic prompts: B=8 seeds 1000 and 5000 (16384 tokens), B=1 seeds 1000-4000
(4096), B=2 seed 6000 (2048): 0 mismatches, 0 exact ties. Re-run on the final source (B=8 seed 3000, B=1 seed 5000,
B=2 seed 7000; 11264 tokens): 0 mismatches.

The switch choices made on seed 1000 hold on other prompt sets: reverting five of them on seeds 2000 and 6000
costs the same amounts within 2 us (journal, ~24.5 agent-h). The prompt set moves the absolute time, not the ranking.
The extension's B=8 changes (s0052-s0056) as a whole: -32 / -30 / -30 us on seeds 1000 / 2000 / 5000 (in process).

## Where warps wait (ncu warp-stall sampling, one generate() launch, ~24.5 agent-h)
Cycles per issued instruction by reason, B=1: block barrier 5.35 of ~12 (45 %: warps idle while one thread polls the
grid barrier or other warps finish), fixed-latency waits 1.74, L2/HBM loads 1.53, shared-memory loads 0.77,
instruction fetch 0.44 (4 %). B=8 (~7 per issue): barrier 1.97, waits 1.52, loads 0.59, fetch 0.32. The kernel is
~290 KB of code per instance, but i-cache misses are not what makes unrelated code changes cost 10-50 us.

## Load balance across SMs is card-dependent; the scored card may not reward it (~27.5-28.5 agent-h)
A dynamic tail for the head (a block's last tiles claimed from a global counter by whichever warp is free) measured
on cards 1-3, track B sketch (SDYN, on a base that already carried another change to shared code): B=1 -6..-18,
B=8 -6..-7 us; on the evaluation card (s0058) the same build was B=1 +2, B=8 +19 over HTTP and +1 / +14 in process.
Once the shared-code change was undone, SDYN was +7..+10 / +38..+46 on cards 2 and 3 as well: the earlier "gain"
was allocation won back. In track A (exact head, HDYN) the profile shows what balancing can give at most: the
head's 152 blocks already stream at ~7.3 TB/s together, the slowest block finishes 9.5 us after the median, and
claiming the last 8 % of tiles brings that to ~4 us (head -4 us at B=1, -1 at B=8).
What it allows: balancing is worth a few us per step here, not more. What it does not: a same-card A/B on cards
1-3 is not a measurement of card 0 when the effect is about per-SM bandwidth; and an A/B of a switch is only as
good as the base it was measured on -- check that the base is SASS-identical to the last submission.

## Head-only summation order is invisible to greedy tokens (s0054, ~26.5 agent-h)
The B >= 3 instance's exact head sums each rowtile as 4 K segments, the B <= 2 instance in one chain: head logits
of a sequence alone and batched differ by up to 5.5e-5 (presubmit's batch check), but greedy tokens alone vs batched
are identical for 8 x 1024 tokens (tools/study/alone_vs_batch.py) and http_tokens stayed 8/8. Unlike an order change
inside the layers (NSEG_QKV_F: alone vs batch diverged, mean prefix 0.20), a head-only difference does not feed back
into the hidden state; it can only flip a near-tie argmax (top-2 within ~1e-5), which did not occur.

## Per-SM streaming speed differs by ~10 %, stably (bench/smbw.cu, pod epoch 8, ~29.5 agent-h)
All 152 SMs stream their own 29.5 MB through the engine's ring shape (8 warps x 3 stages x 4.5 KB bulk copies) at
once: per-SM time min / median / max on card 1 499 / 550 / 603 us, card 2 460 / 555 / 608, card 3 491 / 552 / 601
(max/median 1.09-1.10; the kernel ends with the slowest SM: 7.3 TB/s, while the median SM alone moves 53 GB/s,
~8.1 TB/s over 152). The per-SM times repeat within a card (correlation 0.94-0.97 between launches), are shared by
the two SMs of a TPC (0.92) and only partly across cards (0.56 / 0.68 / 0.86). Blocks of a 152-block cooperative
launch land on fixed SMs (block 0 -> SM 150, 4 -> 0, 6 -> 16, ...; the same in every launch).
In the engine (tools/study/blockspeed.py, B=8 / B=1, pos 600, assuming the engine's blocks map like the benchmark's)
the per-block work of the phases spreads less (E stream at B=8: max/median 1.05) and does not follow the SM's
stream speed (correlation E -0.09, F -0.34, B=8 sketch 0.13); only the B=1 sketch stream does somewhat (0.49).
What it allows: a pure stream could gain up to ~8 % from per-SM weighting or dynamic balancing. What it does not:
the engine's phases are not limited by that spread (consistent with HDYN's small head gain), and a static
weighting learned on one card would only partly fit another.

## At B=8 the expert phases are partly compute-bound through the ring (~30 agent-h)
With the routing pinned to a fixed union of 20 experts per layer (study switch ROUTE_FIX, so that consume-free
builds route the same), B=8 pos 600, profiling build, per layer: E's stream 31.7 us with the consume, 23.9 us without
(NOCONSUME: items loaded, nothing computed) -- 176 MB at 7.4 TB/s, the stream rate; E phase 34.2 -> 26.0, F 21.6 ->
17.1, A 10.7 -> 9.1, C 7.6 -> 6.4. About 15 us per layer (~540 us of the 2991 us step) is consume time that the
ring does not hide. Skipping only the scaled e2m1 conversions (NOCVT) gives E -1.2 us, only the B-fragment loads
(FP4_NOB) -0.5: no single part dominates. A ring model (slot reissued only after its 9 items are consumed) does
not explain it either: bench/smbw2.cu (the per-SM stream with an emulated consume of 60 / 100 cycles per item)
loses only 4-6 % with the engine's 3 x 9 ring; finer rings of the same capacity are much worse (9 x 3: 5.97 /
5.06 TB/s; 5 x 5: 6.94 / 6.69), per-stage overheads dominating. The likely cause is issue contention among the 8
consuming warps with the real instruction mix (~40 per item and warp: 16 scaled conversions, 4 mma, 3 shared
loads, address and loop work), which a spin loop does not reproduce.
What it allows: fewer instructions per weight byte in the fp4 consume is where B=8 time is (up to ~8 us per layer
in E, ~4 in F). What it does not: a ring-geometry change (measured worse), or a cheaper conversion alone.
