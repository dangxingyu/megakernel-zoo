# Plan (run s11-north-h100-b1), final at agent-hour ~13.2

Board: track B best B=1 s0022 2040 (final source s0025 2050, same behaviour), best B=8 s0023 5320; track A s0024
2501 / 6493. Inherited: B 2150 / 6050, A 2540 / 6900.

Remaining losses (gap.md "Final state") and why they stay:
1. B=1 hand-overs (norm 1.5, C tail + release 2.5, routing hop ~1.3 us/layer): every way of filling them with
   traffic beyond the small barrier-wait prefetches lengthened the chain's own round trips; union-timing tricks lost.
   Open lead: the norm's first x load (1.2 us vs 0.33 for a repeat load); causes ruled out are listed in the journal.
2. B=1 stage B tail (ordinary blocks' o_proj): share constants, half o_proj in stage C and two-quanta issue all lost.
   Next mechanism: a K-split of o_proj across block classes with a fixed-order combine.
3. B=8: stage-C spread ~2 us between equally loaded blocks (uneven SMs at the smaller shared-memory allocation):
   speed-aware row-tile or share assignment.
