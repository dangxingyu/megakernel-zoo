# Next run: where to start (written at the end of s11; engine = s0029, 7170 / 7652 us over HTTP)

## Where the time is (B=1, scored contexts 64-1088; gap.md)
- Main kernel ~7005-7030 us: 12-bit GEMM streams ~5.3 ms of bytes, ~5.5 us per GEMM over its bytes (253 GEMMs:
  ramp, slowest warp, cross-block hand-over, epilogue, barrier), recurrent op 9.3 x 48, attention 15-17 x 16,
  escapes ~178 us. Head kernel ~175.5 us (133 us of bytes; floor, F23). Serving path: free (F26).

## The constraint that decides everything (F10, F22, F24)
- ptxas compiles the main kernel as one unit. Any edit to the top-level loop / barrier / gemm_phase re-schedules every
  subroutine including the 12-bit hot loop (1108 of 2342 SASS lines for a one-flag change) and has cost +27..+150 us.
- Edits inside an op's own __noinline__ function are milder (hot loop: 28 lines changed) but still draws: one win
  (LM_VOLC, -29 us over HTTP) against fifteen losers (+1..+61).
- -rdc (separate compilation) is not an isolation: the hot loop then spills 148 B and the pass is +1266 us.
- So the first job of a next run is to make the hot loop's machine code stable under edits elsewhere, or at least
  visible: tools/hotloop_hash.sh prints a hash of gemm_stream_w12e<1,4>'s SASS for the build.sh flags (+ extra
  flags); report it with every A/B. Seen so far: s0027 (LM_VOLC) changed it (2342 -> 2345 instructions), so even
  s0027's gain may be a re-schedule; head-kernel edits do not change it. Textual order does not isolate it (an edit
  in lmix_work, defined after the loop, changed 28 of its lines). Remaining candidates: a frozen cubin for the stream
  path (only if the kernel can stay persistent), or accepting the draw and searching with a fixed protocol (below).

- The strongest cheap lever found late: ptxas --register-usage-level (F28). Level 10 (in build.sh, MK_PTXAS_EXTRA
  overrides) gave -46 us over HTTP on the same source; -disable-optimizer-consts on top -5 / -29 (s0029). Re-sweep levels 5..10 after every source change.

## Measurement protocol that worked (F13)
- nsys medians of the main / head kernel per launch (tools/main_nsys.sh, env_nsys.sh, flag_draws.sh, head_nsys.sh),
  the variant and its base built in the same job on the same card, 3 rounds at B=1, 2 rounds at NSTEP=1024 (the scored
  contexts) and at B=8. Cards differ by up to ~50 us in level; single rounds mislead (DR3: -11 in a screen, +5 after).
- Stamped builds (LMIX_PROF, SK_PROF, ops_ctx.sh) perturb code generation: use them for phase shapes, not for
  effects under ~10 us.
- Exactness before every submission: tools/sub_check.sh (sketch, bitcheck vs track A, served == offline) and
  tools/build_check.sh; tools/robust.sh and tools/stress.py for the final state.

## Measured and closed (do not redo without a new idea)
- Prefetch tables at B=1 and B=8 (converged), placement fillers (F20), instruction-cache warm-up (F21), cluster /
  DSMEM exchange for the recurrent norm (F25, exact, net zero), escape regrouping inside cut-structure classes (F19,
  -14.5 us at most, needs an epilogue map and a norm restructure), column power-of-two scaling (F27, exact, ~-3 us;
  tools/colscale_api.patch), a tighter group-wise sketch bound (F23), build-option draws (F22), HTTP overhead (F26).

## Open, with the number that decides it
- Per-GEMM fixed cost (~5.5 us x 253): needs main-loop / barrier work, i.e. the isolation above first.
- Recurrent op (9.3 us x 48) and attention (15-17 us x 16): latency chains; their own functions can be edited (milder
  draws), but the attention op has rejected every edit tried (+12..+42).
- Rounding-changing ideas (deferred escapes, GEMM rebalancing, attention re-chunking): gate risk ~40 % per change
  (F12); only with a plan for the long-context gate.
