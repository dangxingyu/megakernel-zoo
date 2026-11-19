# Plan (rewritten at agent-hour 23.2; previous rewrite at 21.5)

Best and final engine: s0027 track B 7221 / 7698 us (HTTP B=1 / B=8), new best on the board; engine/ is byte-identical
to the evaluated s0027 source. Inherited 7668 / 8114. Track A (engine/TRACK = A) buildable.

## Since 21.5
- Found that edits inside an op's own function are mild draws (a never-taken block in lmix_work: -0.4 / +3 us), unlike
  top-level edits (F22). Kept: LM_VOLC (recurrent op without its entry spill reloads): s0027 -29 us over HTTP.
- Measured and dropped (journal, F24, F25): ATT_VOLC +42, ATT_PFV +12, LM_VOLC2 +61, cluster launch + DSMEM barrier +41,
  cluster launch + remote-store exchange +6 / 0 (exact but net zero), state prefetch one barrier earlier +1.5.
- The recurrent op's B=8 form (lmix_tri) keeps its inputs in local memory across the IEEE division's slow-path call;
  not the LM_VOLC pattern; B=8 only. Not attempted.

## Remaining time (~6.8 h)
- Nothing with an expected gain above the evaluation's resolution is open: host tables are converged at B=1 and B=8,
  the head is at its byte floor (F23), and main-kernel edits are draws (F22, F24) whose losers outnumber winners.
- Keep engine/ = s0027; finish the record (summary, findings F19-F25, gap).
