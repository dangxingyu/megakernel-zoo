# Where a step's time goes (track B unless stated)

Step time = floor + named losses, in us, each with the measurement it comes from. Updated after every profile.

## Floor (bench/floor.cu, e0007, card 7: 132 x 512 threads, ld.global.L1::no_allocate, cold L2)
- one sweep of 1.07-4.8 GB: 3118-3151 GB/s (the card's practical rate; 94% of the nominal 3350)
- phases of 16 / 39 / 89 MB, one launch each: 2485-2613 / 2842-2890 / 2942-2976 GB/s
- 49 phases of 16 / 39 / 89 MB with a grid barrier between them: 2359 / 2748 / 2876 GB/s (6.88 / 14.16 / 30.83 us per phase)
  -> a phase boundary costs ~1.5 us of ramp-down/up at these sizes even with nothing else to do.
- engine's stage-B loop alone (bench/stream_bench.cu, W12 decode + mma): 768 tiles (37.9 MB, one B=1 gate|up)
  13.6 us = 2784 GB/s; 3072 tiles 2963 GB/s.

## Bytes per step (track B, as the engine reads them)
per layer: q|k|v 15.81 MB, router 0.40, o_proj 12.62, one expert 7.12 (gate|up 4.74 + down 2.38).
- B=1: 48 x (15.81 + 0.40 + 12.62 + 8 x 7.12) + layer 0 o_proj 12.6 + sketch head 541 = 4671 MB -> 1492 us at 3130 GB/s
- B=8 (union U experts per layer, ~27.8 on real text): 48 x (28.8 + 7.12 U) + 554 -> 11.44 GB at U = 27.8 -> 3655 us

## Inherited engine on this pod (e0004, card 2, in process)
| | B=1 | B=8 |
|---|---|---|
| generate, ctx 64-320 | 2126 | 5916 |
| serve (prompt 32, 96 tokens) | 2119 | 5370 (another text: union differs) |
| generate ctx 800-992 | 2170 | 6468 |
| floor (bytes / 3130 GB/s) | 1492 | ~3655 (U = 27.8) |
| loss | ~634 | ~2260 |

## Skeleton (e0008: -DABL_NOSTREAM, every weight quantum returns at once; sketch head still streams)
| build | B=1 step | B=1 gen | B=8 step | B=8 gen |
|---|---|---|---|---|
| no weight streaming | 1044 | 1238 | 2456 | 2658 |
| (not yet read: no attention, both) | | | | |
Skeleton = 1044 us/step at B=1 (21 us per layer); stream (1492) + skeleton (1044) = 2536 vs 2126 measured: only
~410 us of the skeleton overlaps the stream.

## Per-layer timeline, B=1 (e0009, -DTRACE build, tools/trace.py; us after the first block leaves barrier C)
layer 43.4 us (floor for its 85.8 MB: 27.4 us)
- 0 -> 1.7      norm (every block reads the residual: HBM idle)
- 1.7 -> 9.4    stage A (q|k|v + router, 16.2 MB: 5.2 us at 3.13 TB/s); router rows stored +4.85 (last +5.5)
- -> 10.5       union seen by the blocks (leader: poll rows, top-8, publish; blocks: poll)
- 10.5 -> 28.5  gate|up (37.9 MB: 12.1 us); median block done +25.6, last +28.5 (tail 2.9 us)
- -> 33.2       o_proj after a block's gate|up share (12.6 MB), last block at +33.2
- 33.2 -> 41.9  down (19 MB: 6.1 us), C quanta median 8.0 us
- -> 43.4       residual add 0.75 + signal + release
Losses per layer (named): norm 1.7, stage A ramp + routing hop 3.6, gate|up tail 2.9 + rate, o_proj tail into C,
down at 2.4 TB/s 2, residual/barrier 1.5.
B=8: layer 116.5 us (bytes ~226 MB: 72 us): norm 4.4, A 7.8 (A end +13.6), gate|up done median +72.2 / last +77.0
(58.6 us for 132 MB = 2.25 TB/s), C quanta 30.4 us (down 66 MB = 2.2 TB/s), residual 0.8, C->next 37.6.

## Update at s0005 (agent-hour ~3.8): trace e0067 (-DTRACE build, us after the first release from barrier C)
B=1, layer 40.75 (bytes 27.4 at 3.13 TB/s):
| span | us | bytes | at 3.1 TB/s | loss |
|---|---|---|---|---|
| norm (HBM idle) | 1.6 | - | - | 1.6 |
| stage A + routing (union seen by ordinary blocks +8.2) | 6.7 | 16.2 MB | 5.2 | 1.5 |
| stage B: gate|up + o_proj (ordinary blocks' o_proj done +28.0, stage C starts +28.9) | 20.6 | 50.5 MB | 16.3 | 4.3 |
| stage C + residual (median block +37.2) | 8.3 | 19.0 MB | 6.1 | 2.2 |
| C tail + signal + release (next layer +40.75) | 3.5 | - | - | 3.5 |
Ordinary block profile (PROFBLK=50, e0071): gate|up quanta 13.2 us for ~6 tiles = 2.2 us per 3 KB quantum per warp:
every warp waits a full saturated-queue round trip per quantum (Little: 6.3 MB in flight / 3 TB/s = 2.1 us).
B=8 (short context, union 33.8 on that text), layer 107.2 (bytes ~269 MB: 89.7 at 3.0 TB/s): norm 4.4 (x loads 2.0,
sums 1.0, rest 1.2: e0052), stage A 7.2 (5.4), stage B 60 (57.6), B->C 1.7, stage C 27.8 (26.8), residual 0.8,
C tail + release 4.3. At ctx ~830: layer 131.8; attention blocks busy until +60.5.

## Update at s0017 (agent-hour ~8): trace e0117
Board: track B 2101 / 5392 (B=1 / B=8 over HTTP), track A 2517 / 6526.
B=1 (ctx ~830) layer 41.8 us: norm 1.5 | stage A + routing to +8.3 | gate|up to +23.6 (ordinary) | o_proj to +29.0 |
stage C quanta to +37.6 (median; slowest +38.7) | residual +38.3 | release of the next layer +41.8. Named losses per
layer: norm 1.5, stage A/routing ~1.6, stage B ~4.5 (attention blocks busy to +18.9; o_proj 5.4 us after
gate|up), stage C ~1.5, C tail + hand-over 3.4 (of it ~2.3 us from the last residual write to the release; the C->A
hop is 1.0 us with the batch-1 bulk prefetch, 0.5 without).
B=8 (ctx ~95) layer 105.1 (was 116.5 inherited): norm 3.45 (was 4.4), stage A 7.1, stage B 57.7 us for ~170 MB,
stage C 29 us (2.1 us spread between blocks), residual 0.8, C tail + hand-over 4.2. At ctx ~830: 131 us/layer
(another text: larger unions; attention blocks busy to +43).
Unexplained: the norm's x loads (1.2 us B=1 / 1.9 us B=8 per layer, 0.35 / 0.5 in isolation).

## Final state (s0022/s0023, agent-hour ~12): trace e0134 (s0021 source) and the evaluator's figures
B=1: http 2040-2045 us/token, in process 2033 (generate, ctx 64-320); floor for the 4.66 GB a greedy step reads
(12-bit weights, sketch head minus duplicate tiles) at 3.13 TB/s: ~1490 us -> 1.37x the floor (inherited: 1.44x).
Per layer (trace build, ctx ~95) 39.6 us vs 27.4 of bytes:
| span | us | named loss |
|---|---|---|
| norm (x loads ~1.2 us unexplained, 0.35 in isolation) | 1.5 | 1.5 |
| stage A + routing (union seen +8.7; the routing chain is now the longer of the two) | 7.2 | ~2 |
| stage B gate|up + o_proj (ordinary blocks: +8.7 -> +28.6; attention blocks busy to +16) | 19.9 | ~3.6 |
| stage C (quanta 6.8 us for 19 MB) + residual | 7.5 | ~1.4 |
| last residual -> next release | 2.5 | 2.5 |
B=8: http 5320-5391, layer ~105 us at short contexts against ~90 at the stream rate: norm 3.45, stage A excess
~1.7, B->C ~1.5, stage-C spread 2.1, hand-over ~2.1; at long contexts the attention blocks are busy for ~40 us.
