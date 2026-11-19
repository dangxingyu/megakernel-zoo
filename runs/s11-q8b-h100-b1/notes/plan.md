# Plan (rewritten after s0007, run clock ~1.3 h; my session ~8 h)

Done: L2 staging in the attention window (s0003, -64 in process) and at phase ends (s0005 -79, s0006 B=8 -87),
layer-0 table (s0007, -16 / -20), online contract (s0008 pending, -44). Track A: s0002, s0004 (5227 http).
Head sketch being ported by the subagent (var/sketch2; -50 expected on the greedy path).

Projected B=1 after sketch: ~3905 us/token over HTTP on the 64..1088 workload. Floor: 11.0 GB at 3.12 TB/s ~3530
(and L2 hits can exceed it during staged phases). Gap ~375 us.

Three largest losses now:
1. Per-layer excess ~4.4 us x 36 (~160 us): attention window not fully used (attention body grows to ~10.6 us at
   pos 900 while staging is capped at 9 x 16 KB per idle block), o phase wait (~4 us at pos 192: the cold o tiles of
   the attention blocks finish last; STG_OA did not help), barrier latencies.
   a) Staging amount as a function of context (window length): sweep running (e0069, per-bin).
   b) Shorter window at long context: attention over position chunks with fixed-order merge (big change, numerics
      re-drawn; worth ~45 us at B=1 before staging, less now).
2. Attention growth with context (~90 us B=1, more at B=8): same mechanisms as 1b; B=8 attention-window staging
   with small amounts (e0070).
3. Head + embedding (~20-30 us over the head's bytes) and the sketch's own overheads (subagent: pass 2 ~8 us,
   x preparation ~17 us at B=8, unused stash of the 12-bit head).
Track A: carry the 3+ sequence tail staging and the online contract over (separate submissions) when card 0 idles.
