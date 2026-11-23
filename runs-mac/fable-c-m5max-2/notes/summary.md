# fable-c-m5max-2: summary

Final source: the workspace's `engine/` (= **s0011 / s0013**): **8105 / 8096 us per token synthetic, 5124 / 5142
natural** in two evaluations = **1.70x / 1.62x** oMLX oQ4e + DSpark (13767 / 8307), each started after 7 idle minutes. The inherited engine (run 1's s0014 source)
in the same machine state (s0008): 8985 / 5331. **This run is worth -9.8% on synthetic text and -3.9% on natural
text.** Two later evaluations of the same engine, as the machine drifted slower over the last hour (16-row cycles
54.5 and 57.3 ms instead of 53), gave 8203 / 5165 and 8333 / 5300, and a last one after 16 idle minutes (cycles 49.6 ms) 8040 / 5143: the headline
follows the machine's state by about 3% (six evaluations of the final source: 8040, 8061, 8096, 8105, 8203, 8333 and 5124 .. 5300). Every gate passes; the served tokens are the engine's own step-by-step greedy tokens. One-token step about
13100 us (inherited 13946), greedy loop about 12360 us.

Control with DSpark drafts only (s0012: the final source with copy proposals off, same machine state): **8808
synthetic (1.56x), 6847 natural (1.21x)**; run 1's control was 9799 / 7437. The copy proposals are worth 8% on
synthetic and 25% on natural text.

All submissions compute bit-identical logits (numeric p99 0.1013, the s0014 draw) and decode the same continuations.
**The machine was in a slower state for s0001-s0005** (16-row cycles 56-62 ms in the HTTP stage, against 50-53 ms for
s0007-s0011, whatever the engine; the inherited engine measured 9394 / 5637 then and 8985 / 5331 later). Compare
s0001-s0005 among themselves and s0007-s0013 among themselves, not across. The slow state is what a long session of
in-process tests leaves behind (s0014 hit it again after 25 minutes of A/Bs and 7 idle minutes; 15 idle minutes
cleared it).

## What moved the number (HTTP, us per token: synthetic / natural)
| sub | change | synthetic | natural | tokens/event |
|---|---|---|---|---|
| s0001 | the inherited engine, unchanged | 9394 | 5637 | 2.90 / 7.21 |
| s0002 | lossless 12-bit dense matrices; NVFP4 decode by a table per code byte; draft chained behind the verify pass | 9031 | 5427 | 2.90 / 7.21 |
| s0003 | draft depth from an online calibration and the request's tokens per ms | 8550 | 5425 | 2.72 / 6.72 |
| s0004 | the same source again | 8528 | 5378 | 2.72 / 6.72 |
| s0005 | control: copy proposals off (DSpark drafts only) | 9118 | 7120 | 2.35 / 3.80 |
| s0006 | single-pair branch in the expert slots of 2-3 row passes; warm start | 8248 | 5863 | 2.72 / 6.72 |
| s0007 | the same source, cool start (machine in a faster state than for s0001-s0005) | 8249 | 5283 | 2.72 / 6.72 |
| s0008 | the inherited engine (s0001 source) again, same state as s0007 | 8985 | 5331 | 2.90 / 7.21 |
| s0009 | the final source again | 8241 | 5276 | 2.72 / 6.72 |
| s0010 | control: the s0009 source with copy proposals off | 9021 | 7044 | 2.35 / 3.75 |
| s0011 | expert slots of 4 pairs, pair-count branch, one gate/up row pair per simdgroup | 8105 | 5124 | 2.71 / 6.69 |
| s0012 | control: the final source (s0011) with copy proposals off | 8808 | 6847 | 2.34 / 3.77 |
| s0013 | the final source again | 8096 | 5142 | 2.71 / 6.69 |
| s0014 | copies from 8 matching tokens instead of 5; machine in its slow state | 9252 | 5788 | 2.67 / 6.67 |
| s0015 | the same after 15 idle minutes: a wash, reverted | 8165 | 5100 | 2.67 / 6.66 |
| s0016 | the workspace as left (s0011's engine), 12-bit GEMV in a rewritten form | 8203 | 5165 | 2.71 / 6.69 |
| s0017 | the same with the GEMV in its s0011 text; machine drifting slower (cycles 57 ms) | 8333 | 5300 | 2.72 / 6.69 |
| s0018 | the final source after 16 idle minutes (cycles 49.6 ms) | 8040 | 5143 | 2.71 / 6.69 |
| s0019 | the final source again, a few minutes later (cycles 49.7 ms) | 8061 | 5134 | 2.71 / 6.70 |

1. **Lossless 12-bit code for the dense BF16 matrices** (q | gate | rel | sink, k, v, o of every layer: 2.6 GB read per
   pass as BF16). 6 bytes per 4 weights: sign | mantissa bytes and exponent nibbles relative to a per-row window of 16
   exponents; the kernel rebuilds each weight's stored 16 bits and runs the same fma chain, so every sum keeps its
   order and its bits (tests/bitref.py; `pack12` asserts the round trip at load). 19% of the rows have a weight below
   their window (8.8e-5 of the weights) and stay plain BF16 in a side array. The two largest matrices: 4.4 -> 3.2 ms
   per one-row pass. The decode keeps up with the memory (530 GB/s on the packed bytes).
2. **NVFP4 decode by a 256-entry table per code byte** (two weights per lookup): -7% on the experts of a 16-row pass,
   nothing below 8 rows. Decoding by arithmetic instead of a table is 60-90% slower: the expert kernels are
   compute-bound from 3 rows up.
3. **The draft chained behind the verify pass** in one traced graph (acceptance counted on the GPU, the next anchor
   taken from the target's token at the last accepted row): one host sync per DSpark cycle instead of 2-3. Worth ~1%:
   a sync costs ~0.2 ms here.
4. **Draft depth from a cost model**: rows are verified while the calibrated probability that the whole prefix is
   accepted stays above (row ms x the request's recent tokens per ms). The calibration is counted per request
   (P(accept | depth, bucket of the draft's probability), with a decay, from a prior), because the draft's
   probabilities mean different things on different text. -5.3% on synthetic text, nothing on natural text (half of
   whose time is 16-row copy passes). Any rule that stops at ~25-35% acceptance is within 1% of it in process,
   including run 1's unconfirmed "0.5, no minimum" (tests/ab.py).

5. **Expert slots of up to 4 (row, expert) pairs** (s0011) with a branch on the slot's pair count and one gate/up row
   pair per simdgroup. Run 1 had found slots of 4 slower than slots of 2; the cause was padded pairs and 16
   accumulators per lane, not the idea: a 16-row loop pass picks 47 distinct experts for 128 pairs, and slots of 2
   read 79 expert-sized blocks at the memory's rate. Experts of a pass 8.8 -> 8.35 ms at 3 rows, 31.3 -> 28.5 at 16;
   same bits (equal output hashes for every layout). -1.7% synthetic, -3.0% natural in the evaluation.

## What the measurements ruled out (details in notes/journal.md)
- **Token trees**: the DSpark head's second-best first token is right at 1-5% of the anchors (12% on the hardest
  prose); a sibling row for it gains 0.0-0.3% in replay. Not built.
- **Better proposals than n-gram copies**: suffix matches shorter than 5 tokens are right 3-14% of the time unless the
  DSpark chain already says the same, and then they add nothing to it.
- **Fewer host round trips** beyond the chained draft: the GPU idles ~0.5 ms per cycle (graph replay + first
  commit); removing that needs the acceptance, the state commit and the row choice on the GPU with a fixed row count,
  and a fixed row count loses about what the idle time costs.
- **Coarser copies for the draft** (it only proposes): at 5 bits the draft loses acceptance on some prose (a packed
  5-bit kernel exists, off); a 4-bit copy cannot serve the exact-argmax short list (its rigorous bound keeps the
  whole vocabulary).
- **Lossless codes for the experts**: NVFP4 codes carry 3.97 of 4 bits (playbook), and decode work is what the expert
  kernels cannot afford.
- **Proposal and policy parameters** with the final kernels: minimum copy match 8 instead of 5 (-0.9% in process,
  +0.8% / -0.6% in the evaluation: reverted), a DSpark veto on copies (stalls loops the draft cannot predict),
  threshold floor, row cost, Markov steps per stage: all within 1%.
- **Kernel work**: transposed relative-bias table, other slot layouts, MLX's command-buffer thresholds, fewer
  dependent dispatches, message gates with fewer stages (same speed, or 1.2 ms slower when a projection moves into a
  576-thread group), attention value sums with more keys per iteration, a lazy state commit: each 0-1% or a loss.
  The message gates (1.8 ms of a one-row pass in situ) and the attention (1.1 ms) are what is left above the bytes;
  I did not find a faster form of either that keeps their sums. The attention's time is the latency of one
  (row, head) threadgroup (skipping two heads of three changes nothing), so sharing reads among heads would not help;
  splitting these kernels into dispatches of small threadgroups is the untried direction.

## Where the engine stands
- Loops (16-row copy passes, half of the natural workload's time): ~3.1 ms per token, bound by the GPU's arithmetic
  (the experts of 16 rows are 81 G multiply-adds in 30 ms).
- Prose (most of the synthetic workload): a cycle of ~19.5 ms (verify of ~2.3 rows 16.3 ms + draft 2.4 ms + ~0.5 ms
  between cycles) commits ~1.75 tokens. The verify pass reads 2.1 GB of dense code + 0.46 GB of 8-bit unembedding +
  ~1.5 GB of experts per row at 450-530 GB/s, plus ~3.4 ms in ~860 dependent small dispatches. What limits this
  regime is the draft head's accuracy on prose (first draft right 60-70%), not the passes.

## Caveats
- Machine state: see above; a warm start costs up to 11% on top (s0006: natural 5863). The general engines'
  machine state is unknown.
- Natural text gains less than synthetic: 75% of its tokens come from 16-row copy passes, which are bound by the
  GPU's arithmetic (experts 28.5 ms against a floor of ~25 with every expert read once); its headline is also the
  median of three prompts.
- Copy (n-gram) proposals are not the DSpark head; the control above is the engine without them.
- The synthetic headline is the median of three different prompts (7932 .. 9480 within s0003).
- generate() is a plain greedy loop of one-row passes; speculation lives in submit / poll.
- The numeric gate's p99 (0.1013 against a limit of 0.1207) is the s0014 draw; nothing in this run changes a sum's
  order, so it did not move.
