# Gap accounting (rewritten at 8.7 agent-hours, engine = s0020/s0021; B=1 now ~47 / ~68 us per plain / gated
# application after s0022-s0024 (loads moved before polls); tests/crit.py = time between the moments the
# LAST block leaves consecutive phases, i.e. the critical path; profiles at context 1000 with a greedy context)

Floor: 5.54 GB/step at B=1 -> ~1.0 ms at the measured 5.4 TB/s streaming rate. We are at ~2.93 ms in process.

## B=1 (in process ~2.92 ms): 24 plain x ~49 us + 24 gated x ~72 us + unembedding ~137 us
Plain application (us): attention-input mix 4.7 | qkv 6.6 (40 MB, ~bandwidth) | attention 10.5 (items ~4.5: K/V
load 2.6, scores/softmax/PV ~2.4; merger ~7.8 of it is on the path: poll + staging 4.3, max/sum step 2.0 (unexplained:
~100 instructions, not i-cache), output 0.8) | o_proj 4.5 (weights preloaded) | MLP-input mix 3.8 | router 1.0 |
routing 4.0 | w13 7.0 (24 MB) | w2 5.7 | w2b 1.5.
(The merger's 2.0 us max/sum step is a profiler artifact, see journal.)
Gated extra ~23 us: gate D (input mix 3.5 incl. the hand-off wait, message mix 1.0, norm 0.85, 72-unit GEMV 2.2),
F/G 2.0, norm in every block 6.5; MLP gate D 9.4, F/G 1.9, norm 4.7.
What does not help here (measured): any extra streaming overlapping the latency chain (L2 prefetch of qkv/w2/K-V,
L1 prefetch of gate D / qkv), splitting gate D, reordering its mixes, register preloads of weights.

## B=8 (in process ~5.5 ms at context 64-564, HTTP 6.08 ms at 64-1088): per application ~128 (plain) / ~150 (gated)
attention 32 (512 items of 128 keys over 152 blocks = 3-4 waves of ~6.3 us + merger 4.5), w13 25 (~40 union entries,
~120 MB: at the L1-limited streaming rate), w2 28 (2.5x its bytes; blocks whose 1-2 union entries carry 3-4 tokens
take ~21 us vs ~17, max ~30: latency noise across blocks sets the phase), o_proj 11 (waits for the last merger), mixes
8-10 each (designated blocks publish rows, every block polls them), routing ~5.
