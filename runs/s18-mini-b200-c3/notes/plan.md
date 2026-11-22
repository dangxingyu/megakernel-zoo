# Plan (track C, direction: acceptance length), rewritten at 2.8 agent-hours

State: s0006 1416 us/token synthetic b1 (s0001 1527), natural 960 (1017), b8 6279; tokens/event 3.07 / 6.61
(2.88 / 5.22). vLLM + DSpark 1544 / 871 / 2711 (3.48 / 5.21 tokens per step).

## Three largest losses (notes/gap.md)
1. The 8-row verify floor (V(1) 3.47 ms; 80% of a synthetic cycle; step() is 3.1 ms, the s17 track-A step 2.74).
   a) port the s17 step improvements into both verify kernels (outside this run's direction: verification cost);
   b) cheaper K=2..3 passes (V(2) - V(1) = 427 us is the first draft's price) - same.
2. Acceptance of DSpark cycles on synthetic text (2.08 tokens per <=8-row cycle, 87% of synthetic time).
   a) trees / chain + leaves in the 8-row kernel: simulated -1.2% (chain+leaves) .. -2.5% (dynamic trees) over the
      best chain rule; needs per-row ancestor masks in the shared-record attention, conv lags from parents, and a
      cache / ring fix-up for accepted side rows: ~5 h, risky, small.
   b) other drafters (Jacobi reuse, vocabulary restriction, FP4 logits, n-gram extensions): measured, no.
3. The 16-row pass (8.57 ms, 41% of natural time): MoE union growth and 48-query attention are what is left;
   one-pass conversions that paid are in; the rest measured neutral or worse.

## Next
Nothing in this direction is worth more than ~1-2% with the current verifier; the large remaining lever is the
verifier's floor (another run's direction). Remaining time: robustness (done: tests/c3/robust16.py, 14 cases
identical), notes, a final re-validation; small measured items only.

## Stop decision (2.9 agent-hours, 9 h on the board)
Last checks: 16-row MoE layouts (W13_ONEWARP / W13_ROLL slower, W13_CONTIG equal), REC2 slower, online calibration
and copying-regime knobs within +-0.5%. B = 8 speculation was costed in detail: sequences that advance by different
amounts need per-row positions in EVERY pass (also the draft-free ones), i.e. the slower 16-row library (+8% at 8
rows) or a new mode in the main kernel; net est. ~-5% natural_b8, ~0 b8, for 6-8 h of kernel work: not started.
No item left in this direction has an expected gain above evaluation noise (~1%) with this verifier, so development
stops here; s0006 / s0007 is the result. The levers for a next run are in summary.md (verifier floor first).
