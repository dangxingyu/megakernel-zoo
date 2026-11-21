# Plan (rewritten at ~1.6 agent-hours; best track A s0004 = 2911 / 5701 us)

Targets: 1.10x vLLM = 3370 (B=1, passed) / 4641 (B=8, not: 5701; vLLM 5105).

## The three largest losses (gap.md, card 1 profiles e0052 at B=8 ctx 500 / 1000, B=1 ctx 500)
1. B=8 experts: w13 + w2 ~50 us per application = 2.4 ms per step, ~200 MB of distinct expert bytes per application
   at ~4 TB/s; w2 (28 us) is slower than w13 (23 us) for half the bytes: it starts only when w13 has finished.
   a) pipeline w13 -> w2 per expert through a work queue (w2 of expert e starts when e's w13 strips are done):
      could hide most of w2 (~-15 us/app = -0.7 ms); cost: a dynamic scheduler in a persistent kernel, 2-3 h, risky.
   b) deeper in-flight loads in w2 (more bytes per SM in flight): measured by a2/b2 in many variants, small.
2. B=8 attention at long context: 25.6 us/app at ctx 1000 after ATT3 (16.7 at 500).
   a) ATT3 (done, s0004: -159 us HTTP). b) one-round key-range pieces with online softmax (a1 s0008 design; ~12-15
   us/app at ctx 1000) but different records than B<=2: batch bitwise equality lost (allowed within tolerance);
   1.5-2 h. c) team-level merge overlapped with the last round: small (critical path = 3 rounds + merge).
3. Loop gate chains: B=1 ~30 us per gated application (24 of them, ~0.7 ms), B=8 ~41.
   a) GLIN (b2, track B, B=1: -61..-69 us): ported by an agent. b) MSGZ (b2, track B, B>=2: -153 us at B=8): agent.
   c) head sketch SKH (track B): agent.
Next for me: merge the agents' items one at a time (each A/B'd on card 1, bitchecked, submitted), then 2b if time.

# Plan v2 (2.9 agent-hours; best track B s0007 = 2862 / 5364; vLLM 3707 / 5105; target 3370 / 4641)
Done since v1: dual engine (a2 lineage at B=1, a1 at B>=2: the structural answer to loss 2 and to most of loss 1, from
the sibling run's s0002 / s0004); SKH on the B=1 engine; GLIN / MSGZ / L0T measured and not taken (findings.md).
Largest losses now: (1) B=8 is 0.72 ms above the 1.10x target (a1 kernel: experts and gate chains, see a1's gap);
(2) B=1 gate chains ~0.72 ms (every attempt to shorten them by precomputing D halves - GLIN, MSGZ, a2's message
precompute - made F/G and the norm slower by more than D saved; unexplained, likely polling contention);
(3) B=1 head ~100 us with SKH (a two-level int6 / int8 sketch could save ~19 us, b2 sketch4_study).
Remaining 2 h: SKH for the a1 head (agent, ~-50 us at B=8 expected from the a2 measurement), track-A record (s0008),
final notes. No new structure in the last hour.
