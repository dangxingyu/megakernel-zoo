# How to use this workspace (Qwen3-32B decode engine, one B200)

## Build and tracks
- `engine/build.sh` builds `engine/build/libmk.so`; the track comes from `MK_TRACK` or `engine/TRACK` (A = BF16
  weights as stored, B = exact 12-bit storage, `-DW12_ONLY`). The tree is left at `TRACK = B` (= s0036, the best
  submission); `MK_TRACK=A` builds the best track-A engine (same machine code as s0033).
- Track-B runtime defaults differ (engine/api.py): large kernel instance only at 7-8 sequences (`MK_BIG_MINB=7`), no
  per-k-step norm (`MK_NORMF_MINB=99`), attention cap `MK_ATT_SMAX=10`, idle-block prefetch `MK_PF_IDLE=131072`.
- `MK_W12_VERIFY=1` decodes every 12-bit weight with the kernel's arithmetic against the checkpoint at load.

## Checks (all must pass before a submission)
- `python tools/kit/bitcheck.py --a . --a-track A --b . --b-track B`: track B bit-identical to track A, both batch
  invariant (B=1 and B=8, 1152 positions).
- `tests/binv.py` (B=2..8 vs B=1 bitwise), `tests/check.py` (vs HF BF16, batch, generate == step), `tests/stress.py`,
  `tests/robust.py` (limits, serving), `tests/serve.py` (served tokens == step+generate), `tests/cmp32.py T NB lib:api
  hf16` (vs HF float32).

## Measuring (run on cards 1-3; compare variants in the same job)
- `tests/wl.sh B src1.cu src2.cu ...`: builds each source and times positions 128/384/640/896 (the serving workload's
  quarter midpoints), alternating, two rounds; the mean predicts the HTTP figure (B=1 within ~20 us; B=8 HTTP varies
  by up to ~100 us between evaluations).
- `tests/knob.sh B "ENV=..." ...`: the same for runtime settings of the tree's build.
- `tests/prof.py B steps pos` (per-op times between barriers; `MK_ATT_FLAG=0` to separate attention and o),
  `tests/prof4.py` (q/k/v -> attention -> o timeline per block), `tests/prof5.py` (GEMM phases), `tests/prof6.py`
  (B<=6 attention combine), `tests/prof_layers.py` (per layer), `tests/serve_time.py` (served cohort vs generate).
- Phase stamps (MKP/MKQ) read %globaltimer without ordering against bar.sync: compare them only within one build.

## Lessons that cost time (journal.md has the numbers)
- Register allocation of the 7-8 sequence instance moves by 30-40 us (B=8) with any code compiled into it; keep
  changes meant for other instances out of it with template flags, and A/B every change in the same job.
- Single-position A/B misled (s0017); use the workload mean.
- The per-SM L2 read rate (~66 GB/s, ~20-40 beside the weight stream; bench/l2bw.cu) limits every phase that reads
  more than a few KB per block from L2 (norm prep, attention combine, track B's o / down inputs at B=8).
- Conditionally assigned register arrays go to local memory; preloads must be unconditional (clamped).
