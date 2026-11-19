# Notes of this run (Flint on one B200)

- `summary.md`: the best submission and its number, what worked, what did not, what to try next (one page).
- `findings.md`: what was found, each point with its measurement; starts with where the engine stands.
- `journal.md`: every step in the order it happened: what was expected, what was measured, what was concluded,
  including everything that did not work.

## The engine

`engine/mk.cu` is one CUDA source: one persistent kernel (two instances: batch 1-2 and batch 3-8) plus the host C
API; `engine/api.py` loads the checkpoint, packs it into tiles and implements the contract; `engine/build.sh`
builds. `engine/TRACK` holds `A` or `B` (the two results are the same source with this switch; `MK_TRACK` overrides
it). Experiment switches are `-D` macros listed at the top of `mk.cu`, passed with `MK_EXTRA="-D..."`;
`MK_PROF=1` builds the profiling library. Kernel sources of the submissions and of the experiments are kept as
`tools/mk_*.cu.bak` (the journal names them). When an engine starts it times itself for about 13 s to place its
hand-over counters and three small buffers (`api._place_counters`, findings 13g; `MK_NOCAL=1` skips it, `stats()`
reports the pages and the timings).

## Tools (all run in the machine: `mk exec -- python tools/X.py`)

Correctness and determinism
- `test_engine.py --check --long --gen --time --w1 --serve`: gates against my own Hugging Face reference
  (`hf_study.py --out /tmp/mkref` first; after every machine replacement), timing, the 64+1024 workload.
- `bcheck.py [steps]`: every batch size 2..8, every sequence, bitwise against the single-sequence run.
- `digest1.py`, `digest_long.py`, `digest.py`: bitwise digests (batch 1 only / to position 1100 / mixed) to prove
  that a change computes the same thing.
- `flake.py [reps]`: the same workloads repeated in one process, must give the same bits (found both races, and in
  the extension a third: the counter hand-over E_CTR, 4 mismatches in 398 batch-8 repetitions; run 200 before a
  submission that touches a hand-over). `flake_b.py [reps] [B ...]`: the same at batch 7, 5, 3 and for a long generate.
- `api_stress.py`, `long_ctx.py`, `wd_test.py`: contract paths, contexts to 2040, the watchdog path (`wd_test.py`
  needs the build with `MK_EXTRA=-DWD_TEST` and `MK_NOCAL=1`).
- `bdbg.py`, `hid_hash.py`, `b1b8.py`, `zdump.py`/`zcmp.py`: where two runs or two batch sizes first differ.

Measurement
- `prof.py --B 1|8 --pos N` (with `MK_PROF=1`, optionally `MK_PROF_FINE=1`): timeline inside the kernel.
  `bt.py`: per-block timestamps of one layer.
- `bench1..22.cu`: the card (loads, L1/shared memory, L2, hand-overs, layouts; 19: exact division by reciprocal,
  20: cost of instructions on a busy SM, 21: shared-memory bank conflicts, 22: streaming rate by number of blocks
  and the order in which blocks are served); each file says what it measures.
- `pos_time.py`: decode time per token as a function of the position (chunks of 128 tokens).
- `gen_long.py [B ...]`: the scored workload in process (prompt 64 from `ids.npy`, 1024 greedy tokens by
  `generate()`): the figure the extension decided on; a build takes 35 s to judge.
- `ab.sh`, `abt.sh`, `absass.sh "FLAGS" ...`: one line per set of compiler flags (spill lines, in-process timing;
  with the batch-8 timeline; with the shape of the mma runs).
- `sass.sh NAME "FLAGS"`, `sass_seg.py`, `sass_show.py`, `sass_spill.py`: disassembly of the batch > 2 kernel:
  runs of HMMA with the instructions in and between them, a range of instructions, where the spill loads are
  (findings 15a).
- Timing studies: build with `MK_EXTRA=-DWHATIF=bits` (engine/mk.cu says which bit removes what; the results are
  wrong on purpose) and run `gen_long.py` (findings 15g).
- `abg.sh "FLAGS" ...` (build + `gen_long.py`, the A/B used for every decision of the last hours) and
  `abd.sh "FLAGS" [seconds]` (the same for ONE set of flags under a time cap: use it for anything that could hang).
  `clk1.sh [B] [MK_CLK_X]`: study build with the cycle-counter stamps and the timeline of one layer.
  Stamps inside the routing are in `tools/mk_rclk.cu.bak` (copy it over `engine/mk.cu` for the study, restore
  after): `MK_CLK_X=3` names the batch > 2 sites, `MK_CLK_X=4` the batch 1-2 sites (findings 15m).
  Routing variants that were measured and dropped: `tools/mk_rt8.cu.bak` (RT8), `tools/mk_rmatch.cu.bak` (RMATCH),
  `tools/mk_sig4.cu.bak` (SIG4). More stamp sites: `tools/mk_rclk5.cu.bak` (`MK_CLK_X=5`: the batch > 2 candidate
  stage in parts), `tools/mk_rclk6.cu.bak` (`MK_CLK_X=6`: merge and union).
- `vfy.sh PART` (PART 0..3, one per card: all checks before a submission, each under a time cap), `chk.sh "FLAGS"`
  (build with flags + the bit checks), `abd2.sh A B` (A/B/A/B on one card).
- `sass_route.py NAME` (after `sass.sh NAME ""`): the routing's candidate stage in the disassembly.
- `sig_mono.cu` (the router's sigmoid as compiled is monotone over all finite float32), `sig_rcp.cu` (the fast
  path of the compiled division written out is bit-equal for all 2^32 inputs).
- `bt_clk.py --B 8 --pos N` (build with `MK_EXTRA=-DMK_CLK`): timeline of one layer of the NORMAL kernel from the
  cycle counter %clock64, no memory operation in the measured path (findings 15l). Prefer it to `bt.py` / `prof.py`.
- `step_cost.py`: what a host-driven step costs over generate() (launch, argmax, host round trip).
- `bench23.cu` (streaming rate by the shape of a round), `bench24.cu` (how much of the L2 survives other traffic),
  `route_overlap.py` (how much of a step's expert set repeats from earlier positions; needs `-DROUTE_DUMP`).
- `serve_long.py`: the scored path run locally (submit/poll, prompt 64, 1024 tokens, several poll cadences).
- `drawsweep.sh`, `drawtime.sh`: register-allocation draws (spill lines and timings of builds with the switches of
  `tools/mk_draw.cu.bak`; findings 13i).
- `padsweep.sh`, `padsweep2.sh`, `offsweep.sh`, `caltest.sh`: how the address of the counters and of the small
  hand-over buffers changes the step (the experiments behind findings 13g; the first three need the switches of
  `tools/mk_pads.cu.bak`).
- `hf_study.py`, `hf_semantics.py`, `spec_study.py`, `early_route_study.py`, `cert_route_study.py`,
  `spec_agree.py`: the checkpoint and its routing (what Hugging Face really rounds, how predictable the experts are).
