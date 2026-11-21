# s15-mini-b200-b1: Inkling-2-Mini on GB200

Score: `http_step_us_b1` (lower is better). Budget 14.0 agent-hours. 10 submissions so far, 8 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 692
- 1.10x the fastest general engine: 3355
- SGLang (PR #51926 image) (general engine): 4329
- SGLang, GB300 recipe flags at TP1 (general engine): 4557
- vLLM (PR #69028, built from source) (general engine): 3692

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: First persistent-kernel engine (gate check)

- status: **incorrect**, no score, at 0.79 agent-hours, quick evaluation, category structure
- hypothesis: track A: checkpoint formats (BF16 dense, NVFP4/MXFP8 experts with E4M3/UE8M0 scales, unquantized BF16 expert inputs via exact f16 scaling). Gate check of the first correct build; expect numeric median ~0.015-0.025 (limit 0.054), top-1 within a point, invariance bitwise; ~8 ms/step at B=1.
- evaluator: incorrect: failed timed_steps
- gates that failed: timed_steps

### #2 s0002: Persistent kernel, epilogue-fused residual mixes (first full eval)

- status: **valid**, http_step_us_b1 = 5496 (B=8: 1.243e+04), at 1.57 agent-hours, parent s0001, category structure
- hypothesis: track A: checkpoint formats (BF16 dense, NVFP4/MXFP8 experts with their E4M3/UE8M0 scales), unquantized BF16 expert inputs (exact f16 scaling of the BF16 values). First valid full evaluation expected: in-kernel 5.2 ms B=1, 10.9 ms B=8; HTTP B=1 ~5.3 ms (slower than vLLM 3.69); numeric median ~0.02 (limit 0.054); timed_steps fixed by a fresh output tensor per step.
- evaluator: valid: 5496 us/step at B=1 and 12427 at B=8 over HTTP (offline (step through the cohort adapter)); in process 5283 us/step at B=1 (step, MBU 0.131)

### #3 s0003: Fix layer-0 mix dots (normalized X0); ballot-based routing

- status: **valid**, http_step_us_b1 = 5200 (B=8: 1.184e+04), at 2.03 agent-hours, parent s0002, category correctness
- hypothesis: track A: checkpoint formats unchanged (BF16 dense, NVFP4/MXFP8 experts with their scales, unquantized BF16 expert inputs). Correctness fix: s0002 computed layer 0's attention-input RMS from the raw embedding's dots, scaling q/k/v ~4% wrong (numeric median 0.040, numeric_long 0.036 vs s0001 0.025/0.020). Expect numeric median back to ~0.025 and KL median ~0.0006; routing via warp reductions/ballots (no divergent shuffles) expected to cut ~5 us per application -> B=1 HTTP ~5.2 ms (from 5.50).
- evaluator: valid: 5200 us/step at B=1 and 11836 at B=8 over HTTP (offline (step through the cohort adapter)); in process 4983 us/step at B=1 (step, MBU 0.139)

### #4 s0004: Attention prefetch, per-warp f16 input prep, spill fixes in build/reduce, head call

- status: **valid**, http_step_us_b1 = 5041 (B=8: 1.098e+04), at 2.64 agent-hours, parent s0003, category structure
- hypothesis: track A: checkpoint formats unchanged (BF16 dense, NVFP4/MXFP8 experts + scales, unquantized BF16 expert inputs). Attention issues its K/V rows and bias-table loads before the q/r prep (5 -> 2 dependent rounds), MoE input f16 conversion one warp per token (B=8: 15.7 -> ~2 us/app), runtime-indexed arrays removed from the input build and partial reduction, unembedding as one call per block. Same-process A/B on the scored workload vs the s0003 build: 5115 -> 5010 us (B=1), 11195 -> 10398 us (B=8); expect HTTP ~5.10 ms B=1, ~10.4 ms B=8.
- evaluator: valid: 5041 us/step at B=1 and 10981 at B=8 over HTTP (offline (step through the cohort adapter)); in process 4846 us/step at B=1 (step, MBU 0.143)

### #5 s0005: Device greedy loop: in-kernel argmax, generate(), submit/poll

- status: **error**, no score, at 2.73 agent-hours, parent s0004, category launch
- hypothesis: track A: formats unchanged. The engine now serves through submit/poll: kernel launches queued back-to-back, argmax in the unembedding epilogue (64-bit max keys, ties to the lowest id like torch.argmax), the token fed to the next step on the device, tokens published to host-mapped memory by the last block. Removes the per-token host round trip and torch argmax. Expect HTTP B=1 5041 -> ~4950 us, B=8 10981 -> ~10300 us; http_tokens identical (local test: submit/poll tokens == step greedy at B=1 and 8).
- evaluator: the engine passed the gates but could not be measured over HTTP: the HTTP server did not come up

### #6 s0006: Device greedy loop (submit/poll), fixed poll before first submit

- status: **valid**, http_step_us_b1 = 4959 (B=8: 1.094e+04), at 2.82 agent-hours, parent s0005, category launch
- hypothesis: track A: formats unchanged. Same as s0005 (in-kernel argmax keys, device token feedback, submit/poll server thread, tokens via host-mapped memory) with poll() safe before the first submit (s0005's server died there). Expect HTTP B=1 5041 -> ~4950 us, B=8 10981 -> ~10300 us.
- evaluator: valid: 4959 us/step at B=1 and 10936 at B=8 over HTTP (online (submit/poll)); in process 4858 us/step at B=1 (step, MBU 0.142)

### #7 s0007: Track-A base for the sketch head (TRACK=A: exact unembedding everywhere)

- status: **valid**, http_step_us_b1 = 4922 (B=8: 1.084e+04), at 3.15 agent-hours, parent s0006, category cleanup
- hypothesis: track A: formats unchanged (BF16 dense, NVFP4/MXFP8 experts + scales, unquantized BF16 expert inputs). Same source as the coming track-B submission with engine/TRACK=A, so the sketch's gain is measured against its own track A. Head refactor only (argmax keys from a shared epilogue, same MMA order): expect within noise of s0006 (4959 / 10936 us).
- evaluator: valid: 4922 us/step at B=1 and 10836 at B=8 over HTTP (online (submit/poll)); in process 4826 us/step at B=1 (step, MBU 0.143)

### #8 s0008: Track B: certified 8-bit unembedding sketch + exact rescoring (greedy path)

- status: **valid**, http_step_us_b1 = 4882 (B=8: 1.081e+04), at 3.23 agent-hours, parent s0007, category memory
- hypothesis: track B: adds a per-row int8 code of the unembedding (s_i = max|w_i|/127) used only on the greedy path (submit/poll, generate); every other weight in the checkpoint formats, step() logits unchanged (bitcheck vs TRACK=A of the same source: bit-identical at B=1 and 8). Pass 1 streams the int8 head (461 MB instead of 922) with exact integer MMAs against two int8 digits of the hidden state; |exact - sketch| <= (||w_i - s_i q_i|| + 2^-9 ||w_i||) ||x|| + ||s_i q_i|| ||x - x~||; strips whose upper bound reaches the best lower bound are rescored with the exact head's own arithmetic. Greedy tokens identical to exact greedy at 11000/11000 local positions. Expect head 147 -> ~105 us (B=1), 157 -> ~120 us (B=8): HTTP about -40 us at both batch sizes vs s0007.
- evaluator: valid: 4882 us/step at B=1 and 10813 at B=8 over HTTP (online (submit/poll)); in process 4786 us/step at B=1 (generate, MBU 0.143)

### #9 s0009: Attention chunk size 64/128/256 chosen per step (kernel templates)

- status: **valid**, http_step_us_b1 = 4942 (B=8: 1.053e+04), at 3.38 agent-hours, parent s0007, category attention
- hypothesis: track A: formats unchanged. Batch 8 at long contexts ran ~7 64-position attention units per block per layer; the host now picks the largest chunk (64/128/256) that still gives every block a unit and launches the matching kernel instantiation (B=1 always uses 64). Same-process A/B on the scored workload vs s0007's source: B=8 10351 -> 10006 us (-3.3%), B=1 4983 -> 5009 us (+0.5%, the 64 kernel is a separate compile). Expect HTTP B=8 10836 -> ~10500 us, B=1 within +-30 us of 4922.
- evaluator: valid: 4942 us/step at B=1 and 10527 at B=8 over HTTP (online (submit/poll)); in process 4852 us/step at B=1 (step, MBU 0.143)

### #10 s0010: Partial-reduction slot lists from constant tables (no serial setup)

- status: **valid**, http_step_us_b1 = 4852 (B=8: 1.023e+04), at 4.01 agent-hours, parent s0009, category cleanup
- hypothesis: track A: formats unchanged. reduce_parts (2 calls per layer application) built its slot list in a serial thread-0 loop behind a block barrier and mapped slots with another loop; both now come from constant-memory tables built at load. Same arithmetic. Same-process A/B on the scored workload vs s0009's source: 4926 -> 4847 us (B=1), 9903 -> 9745 us (B=8); expect HTTP ~4865 us B=1, ~10370 us B=8.
- evaluator: valid: 4852 us/step at B=1 and 10232 at B=8 over HTTP (online (submit/poll)); in process 4758 us/step at B=1 (generate, MBU 0.145)

