# Gap accounting (updated at 3.6 agent-hours; best track A s0009: 4942 / 10527 us HTTP, track B s0008: 4882 / 10813)

Floor: 5.54 GB/step at 8 TB/s nominal = 692 us; our BF16 GEMV loop streams at 7.4 TB/s with many strips per block
(tests/t_gemvbench.py), the NVFP4 loop at ~4.0 TB/s from HBM (tests/t_nvbench.py, 4x-L2 rotating buffer).

B=1, in kernel, positions ~200 (tests/t_prof.py, t_stamp.py; e0091/e0106): ~4.85 ms = 48 applications + head.
Per non-gated application (~85 us):
| piece | us | what it would need | source |
|---|---|---|---|
| prologue (partial reduce, coefs, input build) | ~9 | ~3 (two dependent rounds) | stamps S0..S3 |
| QKV GEMV + barrier | ~10 | 6.8 standalone for 40 MB | S3->B2, bench |
| attention (64-position units, combine) | ~11 | ~5 | B2->B3 |
| O-proj + epilogue (attn conv, MoE dots) + barrier | ~10.5 | ~4 | B3->B4 |
| MoE input build + router + barrier | ~7 | ~3 | B4->B7 |
| routing + f16 input prep | ~6 | ~2 | B7->S5 |
| W13 + barrier | ~10 | 24 MB: ~6 at 4 TB/s | S5->B8 |
| W2 + epilogue (next X, dots) + barrier | ~13 | ~5 | B8->B9 |
Gated applications (24 of 48) add phases A, B, F1, F2: skipping A and F1 outright saves only 250 us/step (B=1),
300 us (B=8) (variants/skipa, A/B e0119-ish): the gates are not the main loss.
Head: 147 us exact; 105 us with the track-B sketch (greedy path).

B=8 (in kernel ~9.4 ms at 64-320, ~11.0 ms at 64-1088 on real text): expert GEMVs ~84 us/application (W13 ~48,
W2 ~36) for ~140 + ~70 MB at real-text unions (~45 experts): ~2.9 TB/s in engine vs 4.0 standalone; vector phases
scale with B (prologue 4.5 rounds of chunk tasks); attention grew with context (fixed by 64/128/256 chunks, s0009).

Named losses, largest first (B=1): (1) per-phase fixed latency of every GEMV phase (~4-6 us each x 4 phases x 48);
(2) attention latency chain (~6 us x 48); (3) prologue/epilogue vector work (~10 us x 48); (4) gated phases
(~10 us x 24). B=8: expert GEMVs below even their standalone rate (~2.5 ms of the step).

Measured dead ends (do not retry): L2 bulk prefetch of the next GEMV's weights (slows the vector work); deeper
per-warp unrolls (U 12-18) for QKV/O/W13 (callee-saved spills; +2..+7%); inlining the strip functions (+15%);
one function per phase (+35%); 256/384 threads per block (+30%); two/four accumulators in the NVFP4 strip (+15%
standalone, nothing in the engine).
