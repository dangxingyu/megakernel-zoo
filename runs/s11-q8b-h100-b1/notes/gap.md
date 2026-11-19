# Gap: step time = floor + named losses (track B, B=1)

## Current (s0014), run clock ~3.5 h
Evaluator s0014: http 3865 (online path), in process w1-like generate 3865, short-context generate 3810.
Bytes per step now: 12-bit layers 10.39 GB - layer-0 q/k/v 0.04 + int8 head sketch 0.62 (+ ~5 candidate tiles)
= ~10.98 GB -> 3519 us at 3.12 TB/s.

| item | us/token | source |
|---|---|---|
| floor: ~10.98 GB at 3.12 TB/s | 3519 | arithmetic, e0002 rate |
| per-layer excess (avg ~2.95 TB/s over a layer at pos 192: 98 us for 289 MB) ~4-5 us x 36 | ~160 | profile e0084-type, after staging |
|   o phase after attention (staging spill-over + o tile 2; barrier wait ~4 us at pos 192) | part of above | e0076/e0086 |
|   gate+up end imbalance (attention blocks last; staging for them did not help) | part of above | e0076, e0088 |
| attention growth 64 -> 1088 (w1 generate - short generate: 3865 - 3810) | ~55 | s0014; a perfect attention is worth only 12-25 (what-if e0083) |
| head sketch pass 2 + x preparation + embedding/argmax | ~20-30 | subagent probes |
| host (online path: http - w1 generate) | ~0 | s0014 |

Skeleton (e0199, -DSKEL: weight-stream loads replaced by zeros, wrong results; staging, stash addresses, sketch
stream, attention, barriers unchanged): 1874 us/step at B=1, 2357 at B=8 (real: 3814 / 4225). The skeleton is mostly
the 12-bit decode + MMA work (~1.6 us per tile per SM, ~735 tiles per SM per step) and the int8 sketch stream; the
step is ~16% of the skeleton above the byte floor, i.e. most of the non-streaming work is hidden behind the stream.
Where it is not: phase starts whose first tile comes from the stash (compute-bound ~2.7 us with one group per warp
of the next tile in flight) - staging the next tile into L2 at the previous phase's end pays there (s0019).

Closed (measured): entropy coding (ALU), int4-7 head sketches, split attention (B=1 neutral, B=8 worse), L2
staging variants beyond the adopted ones, HBM request mechanisms (TMA, bulk prefetch ahead), knob re-tuning except
O2PF (done).

## Earlier (s0003 evaluated), hour ~6
Evaluator s0003: http 4095, in process w1-like step 4087, generate 4046; short-context step 3992.
TIMING profile e0051 (attention staging on, no tail staging): 99.4 us per layer (slope), embedding + head 319 us.

| item | us/step | source |
|---|---|---|
| weight bytes 11.35 GB at 3.12 TB/s (practical stream rate, bw12.cu) | 3638 | e0002 |
| per-layer excess over bytes: 99.4 - 92.8 = 6.6 us x 36 | ~238 | e0051 slope |
|   of which attention window not covered (pos 192: idle blocks wait 7+ us, partly staged now) | ~100-150 | e0051 phase table |
|   o phase after attention (first tile compute-bound from stash 2.8 us, epilogue+barrier) | ~70 | e0051 |
| embedding + lm_head over the head's bytes (319 - 299) | ~20 | time_layers |
| attention growth 64 -> 1088 (w1-like step - short-context step: 4087 - 3992) | ~95 | s0003 |
| host round trip of the adapter (http - w1 generate: 4095 - 4046) | ~50 | s0003 |

Note: L2 hits add bandwidth on top of HBM (aggregate up to ~3.45 TB/s), so the "floor" at 3.12 TB/s is not a hard
floor for a phase whose bytes were partly staged in L2 during idle windows: gate+up now runs below its byte time.

## Initial (s0001, inherited)
http 4162; w1-like step 4152, generate 4114; short-context step 4019; 103.2 us per layer, head+embedding 318 us.
