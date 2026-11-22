# Gap accounting (B=1 unless stated; in process, context ~1000; tests/crit.py = time between the moments the LAST
# block leaves consecutive phases, i.e. the critical path of the dataflow step)

Floor: 5.54 GB/step. Measured streaming with the engine's load pattern (bench/l2pf.cu): 5.4 TB/s at 40 MB
(7.8 us incl. ~1.5 us ramp), 5.65 TB/s at 64 MB; L2-resident 9.1-9.5 TB/s. Practical floor ~990 us/step.
Round-trip latency (bench/lat.cu): 0.2 us quiet, 1.4 us while other SMs stream HBM (any load flavour, also relaxed);
spin-polling by all other threads costs nothing; __threadfence 0.9 / 2.4-3.5 us. Every hand-off on the critical path
costs ~1-1.5 us while weights stream somewhere.

## s0008 candidate (3500 us in process): step = 24 plain x ~60 us + 24 gated x ~83 us + final ~140 us
Plain application (us): attention-input mix 6.2 | qkv 8.0 (40 MB: 7.4 us of it is bytes at 5.4 TB/s) |
attention 11.6 (chunk items ~5.5 incl. q wait, merger poll + merge ~2-5) | o_proj ~5 (preloaded weights) |
MLP-input mix 5.3 | router 2.4 | routing 5-6.5 | w13 8.7 (24 MB: 4.4 us bytes) | w2 7.1 + 1.5 (12 MB: 2.2 us).
Bytes at 5.4 TB/s: ~17 us of ~60; the rest is ~8 hand-offs x ~1.4 us + in-block latency (block reductions ~0.5-1 us
each, smem/sync chains in attention ~3 us, routing ~3 us).
Gated application extra (~23 us): gate D (x-half job: mix + unit vector + 36-unit GEMV) ~10, F/G ~4, norm ~5;
MLP gate D ~9, F/G ~3.6, norm ~3.9 (minus the plain mixes they replace).
Final: unembedding 922 MB in ~137 us (6.7 TB/s).

## B=8 (in process ~6.8 ms; HTTP 7.67 ms): per application ~136-152 us: attention 25 (2-3 waves of 128-key items),
w13 21-31 (union ~40 experts, 106 MB), w2 27-37, mixes ~10 each, gates.
