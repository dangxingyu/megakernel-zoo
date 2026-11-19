# s11-q8b-h100-b1: Qwen3-8B on H100

Score: `http_step_us_b1` (lower is better). Budget 20.0 agent-hours. 25 submissions so far, 24 valid.

Reference lines of this run:

- SGLang (fastest general engine): 6031
- vLLM (general engine): 6782
- inherited engine, track A (s10 s0004): 5380
- inherited engine, track B (s10 s0002): 4160

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: Inherited engine, track B (s10 s0002 device code), re-measured on this pod

- status: **valid**, http_step_us_b1 = 4162 (B=8: 4457), at 0.10 agent-hours, category other
- hypothesis: track B: weights in the inherited lossless 12-bit code with exact power-of-two row/column scales; float32 intermediates; not bit-identical to track A. No change to the inherited source except TRACK=B. Expect http_step_us_b1 ~4140-4160 (operator rematch 4160), B=8 ~4460-4470: the reference for this run's track-B changes.
- evaluator: valid: 4162 us/step at B=1 and 4457 at B=8 over HTTP (offline (step through the cohort adapter)); in process 4014 us/step at B=1 (generate, MBU 1.126)

### #2 s0002: L2 staging of gate+up tiles by idle blocks during attention (track A)

- status: **valid**, http_step_us_b1 = 5363 (B=8: 5684), at 0.77 agent-hours, parent s0001, category memory
- hypothesis: track A (checkpoint formats, no format differs). During attention at 1-2 sequences the idle blocks, after their o stash, bulk-prefetch (cp.async.bulk.prefetch.L2, 16 KB pieces, no eviction hint) one later gate+up tile (128 KB) of their own block into L2; HBM idles there, and the gate+up phase then gets those lines as L2 hits on top of its HBM stream (microbenchmark: 151 MB stream with 25 MB staged 55 -> 48 us). Reads only: logits bit-identical to the inherited track A (bitcheck 1152 positions, B=1 and B=8). Same-card A/B, generate 64..1088, B=1: 5330 -> 5284 us (3 rounds); B=8 unchanged (staging off for 3+ sequences). Expect http_step_us_b1 ~5330 (inherited 5365-5380, -45), B=8 ~5630 unchanged.
- evaluator: valid: 5363 us/step at B=1 and 5684 at B=8 over HTTP (offline (step through the cohort adapter)); in process 5215 us/step at B=1 (generate, MBU 0.867)

### #3 s0003: L2 staging of gate+up tiles by idle blocks during attention (track B)

- status: **valid**, http_step_us_b1 = 4095 (B=8: 4476), at 0.84 agent-hours, parent s0001, category memory
- hypothesis: track B: formats as inherited (lossless 12-bit weights with exact power-of-two scales, float32 intermediates; not bit-identical to track A). Change: during attention at 1-2 sequences the idle blocks bulk-prefetch 9 x 16 KB of later gate+up tiles of their own block into L2 (no eviction hint); the gate+up phase gets them as L2 hits on top of its HBM stream. Reads only: logits bit-identical to the inherited track B (bitcheck 1152 positions, B=1 and B=8). Same-card A/B, generate 64..1088, B=1: 4120 -> 4033 us (3 rounds). Expect in-process w1-like step ~4065 (s0001 4152), http_step_us_b1 ~4075-4100 (s0001 4162; HTTP adds 10-40 us of noise), B=8 unchanged ~4450.
- evaluator: valid: 4095 us/step at B=1 and 4476 at B=8 over HTTP (offline (step through the cohort adapter)); in process 3992 us/step at B=1 (step, MBU 1.134)

### #4 s0004: L2 staging of the next phase's middle tile at each matvec phase end (track A)

- status: **valid**, http_step_us_b1 = 5227 (B=8: 5663), at 0.92 agent-hours, parent s0002, category memory
- hypothesis: track A (checkpoint formats). On top of s0002: at the end of the o, gate+up and down phases (1-2 sequences), behind the loaders' stash and while the grid barrier runs, 4 threads per block bulk-prefetch 4 x 16 KB of the block's middle tile of the next matvec phase into L2 (no hint); that phase then reads them as L2 hits next to its HBM stream. Reads only: bit-identical to s0002's logits. Same-card A/B, generate 64..1088, B=1: 5287 -> 5163 us (3 rounds, +-1). Expect in-process w1-like step ~5200 (s0002 5322), http ~5200-5240 (s0002 5363; HTTP noise +-25), B=8 unchanged.
- evaluator: valid: 5227 us/step at B=1 and 5663 at B=8 over HTTP (offline (step through the cohort adapter)); in process 5102 us/step at B=1 (generate, MBU 0.886)

### #5 s0005: L2 staging of the next phase's middle tile at each matvec phase end (track B)

- status: **valid**, http_step_us_b1 = 4015 (B=8: 4477), at 0.98 agent-hours, parent s0003, category memory
- hypothesis: track B: formats as inherited (lossless 12-bit weights with exact power-of-two scales, float32 intermediates; not bit-identical to track A). On top of s0003: at the end of the o, gate+up and down phases (1-2 sequences), behind the loaders' stash while the grid barrier runs, 3 threads per block bulk-prefetch 3 x 16 KB of the block's middle tile of the next matvec phase into L2. Reads only: logits bit-identical to the inherited track B (bitcheck 1152 positions, B=1 and B=8). Same-card A/B, generate 64..1088, B=1: 4027 -> 3950 (3 rounds, noisier than track A, which measured -124 and gave -117 in s0004). Expect in-process w1-like step ~4010-4030 (s0003 4087), http ~4020-4050 (s0003 4095), B=8 unchanged.
- evaluator: valid: 4015 us/step at B=1 and 4477 at B=8 over HTTP (offline (step through the cohort adapter)); in process 3904 us/step at B=1 (generate, MBU 1.159)

### #6 s0006: Tail L2 staging also with 3-8 sequences (3 x 16 KB per block)

- status: **valid**, http_step_us_b1 = 3998 (B=8: 4415), at 1.09 agent-hours, parent s0005, category memory
- hypothesis: track B: formats as inherited (lossless 12-bit weights with exact power-of-two scales, float32 intermediates; not bit-identical to track A). Change: the end-of-phase L2 staging of s0005 (next phase's middle tile, 16 KB pieces) now also runs with 3+ sequences, 3 pieces per block. Reads only: logits bit-identical to the inherited track B (bitcheck 1152 positions, B=1 and B=8). Same-card A/B, generate 64..1088: B=8 4420 -> 4332, B=4 4287 -> 4186; the B=1 path is unchanged. Expect B=8 http ~4390 (s0005 4477) and in-process w1-like B=8 step ~4370 (4460); http_step_us_b1 ~4015 +- noise.
- evaluator: valid: 3998 us/step at B=1 and 4415 at B=8 over HTTP (offline (step through the cohort adapter)); in process 3903 us/step at B=1 (generate, MBU 1.159)

### #7 s0007: Layer-0 q/k/v from a per-token table built by the kernel's own q/k/v phase

- status: **valid**, http_step_us_b1 = 4008 (B=8: 4369), at 1.20 agent-hours, parent s0006, category structure
- hypothesis: track B: formats as inherited (lossless 12-bit weights, exact power-of-two scales, float32 intermediates; not bit-identical to track A) plus a [151936 x 6144] float32 table (3.7 GB) of layer 0's q/k/v output per token id, built at load (11 s) by this kernel's own embedding + q/k/v phase, 8 ids per launch. Layer 0's input is the embedding row, so that output depends on the token alone; the step copies the row and skips layer 0's q/k/v phase and its grid barrier (37.7 MB of weights). Logits bit-identical to the inherited track B (bitcheck 1152 positions, B=1 and B=8). Same library, table on/off, 3 rounds: B=1 ~-9 us (noisy), B=8 ~-18. Expect in-process w1-like step B=1 ~3998 (s0006 4008), B=8 ~4355 (4373); http within noise.
- evaluator: valid: 4008 us/step at B=1 and 4369 at B=8 over HTTP (offline (step through the cohort adapter)); in process 3894 us/step at B=1 (generate, MBU 1.161)

### #8 s0008: Online submit/poll contract: one device-loop launch per cohort, tokens via host-mapped memory

- status: **valid**, http_step_us_b1 = 3952 (B=8: 4317), at 1.29 agent-hours, parent s0007, category launch
- hypothesis: track B: formats as s0007 (lossless 12-bit weights with exact power-of-two scales, float32 intermediates, layer-0 q/k/v table; not bit-identical to track A). Change: the engine implements submit()/poll(): per cohort of equal-length prompts the prompt goes through step(), then ONE k_generate launch (argmax, feedback, loop on the device) for all new tokens; the kernel stores each step's tokens tagged per cohort into host-mapped memory (8-byte store + system fence), poll() reads them without synchronizing the device. Removes the adapter's per-token torch argmax + launch + host read. step()/generate() logits unchanged (bitcheck bit-identical to the inherited track B, B=1 and 8); served == generate == step+argmax (1024 / 8192 tokens), back-to-back cohorts, input validation. Same build: -44.1 us/token at B=1, -43.6 at B=8. Expect http_step_us_b1 ~3955-3965 (s0007 w1 generate 3954), B=8 ~4310-4320; host syncs per step ~0.
- evaluator: valid: 3952 us/step at B=1 and 4317 at B=8 over HTTP (online (submit/poll)); in process 3890 us/step at B=1 (generate, MBU 1.162)

### #9 s0009: Online submit/poll contract (track A)

- status: **valid**, http_step_us_b1 = 5160 (B=8: 5588), at 1.36 agent-hours, parent s0004, category launch
- hypothesis: track A (checkpoint formats, no format differs). On top of s0004: the engine implements submit()/poll(): prompt through step(), then one k_generate launch per cohort (device argmax and feedback), tokens tagged per cohort in host-mapped memory, poll() never synchronizes the device. Removes the adapter's per-token argmax + launch + host read (~44 us on track B: s0008). step()/generate() device code unchanged (track A bit-identical to s0004's). Expect http_step_us_b1 ~5170-5180 (s0004 w1-like generate 5165, http 5227), B=8 ~5590.
- evaluator: valid: 5160 us/step at B=1 and 5588 at B=8 over HTTP (online (submit/poll)); in process 5103 us/step at B=1 (generate, MBU 0.887)

### #10 s0010: Tail L2 staging also with 3-8 sequences (track A)

- status: **valid**, http_step_us_b1 = 5162 (B=8: 5486), at 1.43 agent-hours, parent s0009, category memory
- hypothesis: track A (checkpoint formats). On top of s0009: the end-of-phase L2 staging (next phase's middle tile, 16 KB pieces) also with 3+ sequences, 3 pieces per block (as s0006 on track B). Reads only: bit-identical to s0009. Same-card A/B, generate 64..1088, B=8: 5578 -> 5481 (2 rounds). Expect B=8 http ~5495 (s0009 5588), B=1 unchanged ~5160 +- noise.
- evaluator: valid: 5162 us/step at B=1 and 5486 at B=8 over HTTP (online (submit/poll)); in process 5101 us/step at B=1 (generate, MBU 0.887)

### #11 s0011: Certified int8 sketch of lm_head + exact rescoring on the greedy/online paths

- status: **valid**, http_step_us_b1 = 3895 (B=8: 4248), at 1.51 agent-hours, parent s0008, category memory
- hypothesis: track B: formats as s0008 (lossless 12-bit weights, exact power-of-two scales, float32 intermediates, layer-0 q/k/v table; not bit-identical to track A) plus an int8 copy of lm_head with per-row scales and certified error constants (625 MB), used only by generate()/the online path: pass 1 bounds every logit from the int8 sketch (x as two exact int8 digits, integer MMA, directed rounding), pass 2 recomputes with the exact 12-bit head's own code every tile whose upper bound reaches the best lower bound (4.9 tiles per step at B=1, 35.6 at B=8, of 9496), so the greedy token is the exact head's argmax. step() logits keep the exact head (bitcheck bit-identical, B=1 and 8). Sketch token == exact argmax at 10,880 teacher-forced positions (incl. all-tiles mode), served == generate == step+argmax. Same build, online path: -53.6 us/token B=1, -52.8 B=8. Expect http_step_us_b1 ~3900 (s0008 3952), B=8 ~4265 (4317).
- evaluator: valid: 3895 us/step at B=1 and 4248 at B=8 over HTTP (online (submit/poll)); in process 3829 us/step at B=1 (generate, MBU 1.167)

### #12 s0012: Token ids outside the vocabulary: no read past the tables, ValueError

- status: **valid**, http_step_us_b1 = 3869 (B=8: 4263), at 1.82 agent-hours, parent s0011, category correctness
- hypothesis: track B: formats as s0011 (lossless 12-bit weights, exact power-of-two scales, float32 intermediates, layer-0 q/k/v table, int8 head sketch on greedy paths; not bit-identical to track A). Correctness only: an id outside [0, 151936) would have read past the embedding and the 3.7 GB layer-0 table; the kernel now uses id 0 for it and sets a host-mapped error word that step() raises as ValueError at the next call (no device sync); generate() and submit() reject such ids on the host. Valid inputs: logits bit-identical (bitcheck vs inherited track B, B=1 and 8), online tests unchanged. Expect timings equal to s0011 within noise (http ~3895, B=8 ~4250).
- evaluator: valid: 3869 us/step at B=1 and 4263 at B=8 over HTTP (online (submit/poll)); in process 3843 us/step at B=1 (generate, MBU 1.161)

### #13 s0013: Head sketch: x preparation with 16-byte loads in the raw fragment layout

- status: **valid**, http_step_us_b1 = 3881 (B=8: 4247), at 1.99 agent-hours, parent s0012, category gemm
- hypothesis: track B: formats as s0012 (lossless 12-bit weights, exact power-of-two scales, float32 intermediates, layer-0 table, int8 head sketch on greedy paths; not bit-identical to track A). Change in the sketch path only: the per-block preparation of x for the int8 passes reads x with 16-byte loads in the raw fragment layout (same exact float math). Tokens unchanged (sketch == exact argmax at 10,880 positions; served == generate == step+argmax); step() bit-identical. One-instance A/B: B=8 -6..-16 us, B=1 ~0. Expect B=8 http ~4245 (s0012 4263, s0011 4248), B=1 ~3870-3895 (noise).
- evaluator: valid: 3881 us/step at B=1 and 4247 at B=8 over HTTP (online (submit/poll)); in process 3851 us/step at B=1 (generate, MBU 1.162)

### #14 s0014: No L2 prefetch of the second tile behind a stashed first tile (O2PF 0)

- status: **valid**, http_step_us_b1 = 3865 (B=8: 4255), at 2.68 agent-hours, parent s0013, category scheduling
- hypothesis: track B: formats as s0013 (lossless 12-bit weights, exact power-of-two scales, float32 intermediates, layer-0 table, int8 head sketch on greedy paths; not bit-identical to track A). Change: the 6 prefetch.global.L2 requests for the rest of a block's second tile, issued while its stashed first tile is multiplied, are dropped: with the L2 staging since s0003 they compete with staged traffic. Hints only: logits bit-identical. Re-tuning sweep on the staged engine, same card, 4 rounds: B=1 -26 us (-21/-50/-11/-23), B=8 +-0. Expect in-process w1 generate B=1 ~3873 (s0013 3899), http ~3855-3880 (noise +-15), B=8 ~4245.
- evaluator: valid: 3865 us/step at B=1 and 4255 at B=8 over HTTP (online (submit/poll)); in process 3810 us/step at B=1 (generate, MBU 1.170)

### #15 s0015: Track A re-tuned for the staged engine: 1.5 staged tiles, 4 stash loads in flight, no o prefetch

- status: **incorrect**, no score, at 3.09 agent-hours, parent s0010, category scheduling
- hypothesis: track A (checkpoint formats, no format differs). On top of s0010: three scheduling constants re-tuned after the L2 staging changed the traffic at phase transitions (track-A-only defaults): attention-window staging 12 x 16 KB (1.5 BF16 tiles) instead of one tile, end-of-phase stash with 4 loads per warp in flight instead of 2, no prefetch of the attention blocks' o tile. Hints/scheduling only: logits bit-identical to s0010. Same-card sweep then combination, 3 rounds: B=1 -32, B=8 -63. Expect http ~5130 (s0010 5162), B=8 ~5425 (5486).
- evaluator: incorrect: failed numeric, tokens, batch, numeric_long
- gates that failed: batch, numeric, numeric_long, tokens

### #16 s0016: Track A re-tuned: 1.5 staged tiles, 4 stash loads in flight (fix of s0015)

- status: **valid**, http_step_us_b1 = 5134 (B=8: 5462), at 3.21 agent-hours, parent s0010, category scheduling
- hypothesis: track A (checkpoint formats, no format differs). On top of s0010: attention-window staging 12 x 16 KB (1.5 BF16 tiles) instead of one tile, end-of-phase stash with 4 loads per warp in flight instead of 2 (track-A-only defaults). s0015 also dropped the o-tile prefetch of the attention blocks, which exposed a latent bug (the stash-count reset sat inside that prefetch's #if; fixed, prefetch kept). Bitcheck before submitting: logits bit-identical to the inherited track A (1152 positions, B=1 and B=8, batch invariance 1.0). Same card, 3 rounds: B=1 -27, B=8 -46. Expect http ~5135 (s0010 5162), B=8 ~5440 (5486).
- evaluator: valid: 5134 us/step at B=1 and 5462 at B=8 over HTTP (online (submit/poll)); in process 5072 us/step at B=1 (generate, MBU 0.892)

### #17 s0017: Exception MMAs read x from shared memory instead of L2 (from a phase's 2nd tile)

- status: **valid**, http_step_us_b1 = 3837 (B=8: 4240), at 3.60 agent-hours, parent s0014, category gemm
- hypothesis: track B: formats as s0014 (lossless 12-bit weights with exact power-of-two scales and exact exception patches, float32 intermediates, layer-0 table, int8 head sketch on greedy paths; not bit-identical to track A). Change: with 1-2 sequences the coordinator's exception MMAs (weights outside the 12-bit window, ~550K) gathered their x values from the global compact copy before each tile's block barrier, an L2 round trip under the full weight stream that every loader warp waited for (what-if without exceptions: B=1 -38 us). From a phase's second tile on they now read the same values from the loaders' shared-memory slices. Same values: logits bit-identical (bitcheck B=1/2/8), online + edge tests pass. Same card, 4 rounds: B=1 -40 us, B=2 -58, B=8 -8. Expect http ~3825 (s0014 3865), B=8 ~4245.
- evaluator: valid: 3837 us/step at B=1 and 4240 at B=8 over HTTP (online (submit/poll)); in process 3775 us/step at B=1 (generate, MBU 1.158)

### #18 s0018: Exception x from shared memory only in the greedy kernel (fixes s0017's step() spill)

- status: **valid**, http_step_us_b1 = 3825 (B=8: 4258), at 3.89 agent-hours, parent s0017, category gemm
- hypothesis: track B: formats as s0017 (lossless 12-bit weights with exact power-of-two scales and exact exception patches, float32 intermediates, layer-0 table, int8 head sketch on greedy paths; not bit-identical to track A). s0017's shared-memory read of exception x values made k_step and k_generate spill 48 B (step() path +44 us in s0017's w1-like step; same-card +69); now compiled only into k_gensk (greedy/online path), all three kernels without spills. Logits bit-identical (bitcheck B=1/8). Same card: step() path +2.6 (noise) vs before s0017, generate path -35.5. Expect http ~3830-3840 (s0017 3837), w1-like step back to ~3990 (s0017 4033).
- evaluator: valid: 3825 us/step at B=1 and 4258 at B=8 over HTTP (online (submit/poll)); in process 3773 us/step at B=1 (generate, MBU 1.178)

### #19 s0019: Tail staging after o targets the block's 3rd gate+up tile instead of the middle one

- status: **valid**, http_step_us_b1 = 3843 (B=8: 4257), at 6.16 agent-hours, parent s0018, category memory
- hypothesis: track B: formats as s0018 (lossless 12-bit weights with exact power-of-two scales and exact exception patches, float32 intermediates, layer-0 table, int8 head sketch on greedy paths; not bit-identical to track A). Change: the 3 x 16 KB L2 staging at the end of the o projection now targets gate+up tile 2 (of 12) of the block instead of tile 6, i.e. the tiles that are streamed right after the stashed first tile, when too few loads are in flight (one group per warp) while the first tile is multiplied from shared memory. Hints only: logits bit-identical (bitcheck B=1/8, both tracks), no spills in the three kernels. Same card, 6 rounds: B=1 -20 us (5 of 6 rounds -20..-26), B=8 -8. Expect http ~3805-3815 (s0018 3825), B=8 ~4250.
- evaluator: valid: 3843 us/step at B=1 and 4257 at B=8 over HTTP (online (submit/poll)); in process 3773 us/step at B=1 (generate, MBU 1.177)

### #20 s0020: Attention window: shorter score-stamp wait, more polite o-stash loads (track B re-tune)

- status: **valid**, http_step_us_b1 = 3813 (B=8: 4253), at 6.81 agent-hours, parent s0019, category scheduling
- hypothesis: track B: formats as s0019 (lossless 12-bit weights with exact power-of-two scales and exact exception patches, float32 intermediates, layer-0 table, int8 head sketch on greedy paths; not bit-identical to track A). Change (1-2 sequences, scheduling hints only): the idle blocks poll at most 8 instead of 16 times for the attention blocks' score-pass stamp before stashing, and issue their first 7 instead of 5 o-stash loads one at a time. Re-tuned after the L2 staging changed the window's traffic. Logits bit-identical (bitcheck B=1/8, both tracks), no spills. Same card: 3 rounds -20 each; 5-round confirmation of both together -11 at B=1 (every round negative), B=8 -5. Expect in-process w1 generate ~3814 (s0019 3825), http within noise (~3800-3840).
- evaluator: valid: 3813 us/step at B=1 and 4253 at B=8 over HTTP (online (submit/poll)); in process 3776 us/step at B=1 (generate, MBU 1.173)

### #21 s0021: Head sketch: rescoring of candidate tiles spread over all blocks (3+ sequences)

- status: **valid**, http_step_us_b1 = 3822 (B=8: 4241), at 9.64 agent-hours, parent s0020, category scheduling
- hypothesis: track B: formats as s0020 (lossless 12-bit weights with exact power-of-two scales and exact exception patches, float32 intermediates, layer-0 table, int8 head sketch on greedy paths; not bit-identical to track A). Change (greedy/online path, 3+ sequences): the sketch's pass 2 rescored each candidate tile on the block that owns it (~36 tiles at B=8, streamed cold, the busiest owner set the pace); now candidates go to a global list (one extra grid barrier) and block b rescores entries b, b+132, ... with the exact head's own code; the argmax reduction (value, lowest row) is unchanged, so the token is the same. Sketch == exact argmax at 10,880 teacher-forced positions with balancing forced on at every B; online tests green; step() bit-identical; no spills. One-instance A/B: B=8 -14, B=1 unchanged path. Expect B=8 http ~4240 (s0020 4253), B=1 within noise (~3800-3840).
- evaluator: valid: 3822 us/step at B=1 and 4241 at B=8 over HTTP (online (submit/poll)); in process 3762 us/step at B=1 (generate, MBU 1.177)

### #22 s0022: 3+ sequences: stage gate+up into L2 after the attention blocks' score pass

- status: **valid**, http_step_us_b1 = 3813 (B=8: 4231), at 12.36 agent-hours, parent s0021, category memory
- hypothesis: track B: formats as s0021 (lossless 12-bit weights with exact power-of-two scales and exact exception patches, float32 intermediates, layer-0 table, int8 head sketch on greedy paths; not bit-identical to track A). Change (3+ sequences, hints only): the idle blocks of the attention phase, after their o stash, wait (bounded poll) for the attention blocks' score-pass stamps (now also written with 3+ sequences) and then bulk-prefetch 3 x 16 KB of a later gate+up tile into L2: the K reads of the score pass are bandwidth-bound, the softmax / V part of the window is not, so the staging fills that part. Logits bit-identical (bitcheck B=1/4/8), online tests green, no spills. Same card, 4 rounds: B=8 -11 (-12/-12/-10/-12), B=4 -30 (earlier sweep); B=1 path unchanged. Expect B=8 http ~4230 (s0021 4241), B=1 within noise.
- evaluator: valid: 3813 us/step at B=1 and 4231 at B=8 over HTTP (online (submit/poll)); in process 3764 us/step at B=1 (generate, MBU 1.178)

### #23 s0023: 3+ sequences: idle blocks' o-tile prefetch deferred until the score pass is done

- status: **valid**, http_step_us_b1 = 3815 (B=8: 4221), at 12.58 agent-hours, parent s0022, category scheduling
- hypothesis: track B: formats as s0022 (lossless 12-bit weights with exact power-of-two scales and exact exception patches, float32 intermediates, layer-0 table, int8 head sketch on greedy paths; not bit-identical to track A). Change (3+ sequences, hints only): the idle blocks' L2 prefetch of their second o tile (9 k-steps at 6+ sequences) now also waits for the attention blocks' score-pass stamps, so it no longer competes with the bandwidth-bound K reads. Logits bit-identical (bitcheck B=1/4/8), online tests green, no spills. Same card, 3 rounds: B=8 -9.4 (-10/-10/-9), B=4 0; B=1 path unchanged. Expect B=8 http ~4222 (s0022 4231.5), B=1 within noise.
- evaluator: valid: 3815 us/step at B=1 and 4221 at B=8 over HTTP (online (submit/poll)); in process 3764 us/step at B=1 (generate, MBU 1.177)

### #24 s0024: 3+ sequences, long contexts: idle blocks pull V (only) into L2 during attention

- status: **valid**, http_step_us_b1 = 3817 (B=8: 4211), at 12.88 agent-hours, parent s0023, category attention
- hypothesis: track B: formats as s0023 (lossless 12-bit weights with exact power-of-two scales and exact exception patches, float32 intermediates, layer-0 table, int8 head sketch on greedy paths; not bit-identical to track A). Change (loads only): beyond the idle blocks' K/V touch limit (256 position tiles x sequences, i.e. past ~512 positions at B=8) they now still pull the layer's V history into L2, which the attention blocks need only after the softmax (the V pass was the longest part of the B=8 body, 5.85 us at pos 576). Logits bit-identical (bitcheck B=1/4/8), online tests green, no spills. Same card, 3 rounds: B=8 -14.5 (-15/-14/-14), B=4 -1; B=1 path unchanged. Expect B=8 http ~4206 (s0023 4220.6), B=1 within noise.
- evaluator: valid: 3817 us/step at B=1 and 4211 at B=8 over HTTP (online (submit/poll)); in process 3766 us/step at B=1 (generate, MBU 1.177)

### #25 s0025: Track A: idle blocks pull V into L2 during attention beyond the touch limit

- status: **valid**, http_step_us_b1 = 5134 (B=8: 5485), at 13.19 agent-hours, parent s0016, category attention
- hypothesis: track A (checkpoint formats, no format differs). On top of s0016 (+ the shared-source changes since, which are compiled out or off for track A, so its device behaviour was s0016's): beyond the idle blocks' K/V touch limit (256 position tiles x sequences, past ~512 positions at B=8) they now pull the layer's V history (only) into L2 for the attention blocks' V pass (as s0024 on track B). Loads only: logits bit-identical to the inherited track A (bitcheck B=1/8). Same card, 4 rounds: B=8 -36 (-53/-18/-28/-42), B=1 ~0. Expect B=8 http ~5430 (s0016 5462), B=1 ~5134 +- noise.
- evaluator: valid: 5134 us/step at B=1 and 5485 at B=8 over HTTP (online (submit/poll)); in process 5073 us/step at B=1 (generate, MBU 0.892)

