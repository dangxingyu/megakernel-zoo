# s10-q8b-h100-f1: Qwen3-8B on H100

Score: `http_step_us_b1` (lower is better). Budget 3.0 agent-hours. 4 submissions so far, 4 valid.

Reference lines of this run:

- SGLang (fastest general engine): 6031
- vLLM (general engine): 6782
- inherited engine, track A (run s2, s0066): fails numeric_long under today's evaluator: 5380
- inherited engine, track B (run s2, s0065): 4160

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: Track A: attention scores stay float32 (no BF16 rounding of q.k)

- status: **valid**, http_step_us_b1 = 5376 (B=8: 5634), at 0.28 agent-hours, category correctness
- hypothesis: Track A. The inherited build rounded each attention score to BF16 twice, bf16(bf16(q.k)*scale) (the HF 'eager' form); the fused attention kernels the general engines and HF's default path run keep scores in float32 (BF16 only for the probabilities and the output). Measured before the change on my own 8x1153 sequences vs HF f32 at positions 128..1152: HF BF16 default median 0.0119, HF BF16 eager 0.0144, engine with 2 roundings 0.0145 (follows eager layer by layer, 1.17-1.28x from layer 1), 1 rounding 0.0134, float32 scores 0.0122 (0.99-1.02x HF default at every layer). Change: one macro at the two score sites (one-block and split path). Expect numeric_long to pass with median ~1.0x HF BF16 (about 0.016, limit 0.032) and a largest error that is a draw from HF BF16's own tail (expect 0.4-0.9, limit 1.068); short-position numeric median down from 1.17x to ~1.0x HF; speed unchanged (local 5233 / 5383 us step B=1 / B=8), http_step_us_b1 ~5380. No sync point added or removed.
- evaluator: valid: 5376 us/step at B=1 and 5634 at B=8 over HTTP (offline (step through the cohort adapter)); in process 5234 us/step at B=1 (generate, MBU 0.864)

### #2 s0002: Track B build of the s0001 source (12-bit weight code, float32 intermediates)

- status: **valid**, http_step_us_b1 = 4136 (B=8: 4461), at 0.68 agent-hours, parent s0001, category memory
- hypothesis: Track B = the s0001 source built without -DTRACK_A. What differs from the track-A build: (1) weights read as a lossless 12-bit code of the checkpoint's BF16 values (sign, 4-bit exponent code, 7-bit mantissa) with exact power-of-two row/column scales and exact patches for out-of-window elements, 3 loads per 4 ksteps instead of 4; (2) residual stream, matvec outputs, SiLU product inputs and q/k norm + RoPE steps are kept in float32 instead of being rounded to BF16 after every operator (K/V cache, probabilities and every matvec input stay BF16); (3) RMSNorm is fused: the matvec input is bf16(w*x) and 1/rms multiplies the outputs. Logits are NOT bit-identical to track A; they are within the evaluator's limits (more accurate: my long-position set median 0.0051 vs track A 0.0122 and HF BF16 0.0119). The s0001 change (score rounding) is a no-op here: verified bit-identical to the inherited s0065 source on 1300 positions at B=1 and B=8 (1.78e9 logits, tests/cmp_exact.py). Expect the inherited numbers: numeric_long median ~0.0074, max ~0.50; in process ~4047 / 4175 us (generate, B=1 / B=8); http_step_us_b1 ~4160, B=8 ~4460.
- evaluator: valid: 4136 us/step at B=1 and 4461 at B=8 over HTTP (offline (step through the cohort adapter)); in process 4040 us/step at B=1 (generate, MBU 1.119)

### #3 s0003: Track A: final source (test-only switches, trace-build fix), same device code as s0001

- status: **valid**, http_step_us_b1 = 5373 (B=8: 5634), at 1.45 agent-hours, parent s0001, category cleanup
- hypothesis: Track A. Since s0001 the source gained two compile-time switches that are off by default (SCORE_SCALE: the score scale as a macro, used by the jitter builds of the tail study; VSB: V loads in flight below 1136 positions, measured no gain) and a one-token fix of the TRACE build. The default build's device code is the same: SASS dump of the track-A library identical to s0001's (42562 lines), preprocessed source equal up to whitespace and host line numbers. Expect every gate exactly as s0001 (numeric_long median 0.015726, max 0.640394 at (1, 952)) and the same speed within the 20-30 us spread between evaluations: http_step_us_b1 ~5376, B=8 ~5634. No sync point added or removed.
- evaluator: valid: 5373 us/step at B=1 and 5634 at B=8 over HTTP (offline (step through the cohort adapter)); in process 5231 us/step at B=1 (generate, MBU 0.864)

### #4 s0004: Track A: idle blocks pull K of all units into L2 first (1-2 sequences)

- status: **valid**, http_step_us_b1 = 5365 (B=8: 5622), at 2.30 agent-hours, parent s0003, category scheduling
- hypothesis: Track A, scheduling only. At the start of attention the idle blocks pull the K/V history into L2 for the attention blocks; the inherited order went unit by unit (K then V of each), so a late unit's K was requested after the early units' V while the score pass needs K first. Now, with 1-2 sequences: K of all units first, units interleaved, then V. Reads only: logits bitwise identical to s0001 (verified on 1300 positions at B=1 and B=8, 1.78e9 logits), so every numeric gate must equal s0001's exactly (numeric_long max 0.640394). Local A/B on 64..1088 positions, B=1, three rounds: 5337.2 / 5337.1 / 5338.9 -> 5329.1 / 5326.2 / 5330.1 us per step (-9, 0.17%); B=2 and B=8 unchanged. Expect http_step_us_b1 about 5365 (s0001 5376.0, s0003 5373.0: the gain is about the size of the spread between evaluations), B=8 ~5634 unchanged. The track-B build's device code is unchanged (SASS identical). No sync point added or removed.
- evaluator: valid: 5365 us/step at B=1 and 5622 at B=8 over HTTP (offline (step through the cohort adapter)); in process 5231 us/step at B=1 (generate, MBU 0.865)

