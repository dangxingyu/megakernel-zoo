# Plan (rewritten 2026-11-17 at ~15.7 agent-h)

State: track B s0038/s0040 HTTP B=1 1719-1722 (best s0031 1715), B=8 3083: 1.276x / 1.111x SGLang -- the 1.10x
claim holds at both batch sizes (h0_step since s0036). Track A s0039 1798 / 3117: 1.22x / 1.098x (B=8 4 us short).

What moved since the ~13 h plan: a bench that reproduces the HTTP client's prompts (bench.py --synth; the scored
B=8 workload routes far more alike than real text) and re-sweeps of same-bits switches on it: HSK_XG (-15),
FTAIL 0 (-10), ATT_BLKW 2 (-11), NORM_PIPE1 (-23.5) for track B; PF_STAGES_W 3 (-19) for track A. Knob values
flip with the register allocation (FTAIL, ATT_BLKW, STEPARG, NORM_PIPE1 all changed sign at some point), and the
two tracks' builds want different values (NORM_PIPE1, PF_STAGES_W are now track-dependent defaults).

Three largest losses now (unchanged in kind): latency-bound A-D phases (~33 us/layer), the head (B=8 ~118 us track
B / ~171 track A), E/F at B=1 (19.8 us/layer vs 7.2 at HBM rate: per-warp stage chains, routing known only at E).
Everything structural tried against them in this run is in gap.md's dead-end list.

Rules kept: same bits across both instances (alone == batch: greedy alone vs batch diverges within tens of tokens
otherwise -- NSEG_QKV_F); judge bit-changing variants on fixed content (bench.py --fixed); one benchmark job at a
time on cards 2/3; one idea per submission.

Remaining ~8 h: no large changes. Keep both tracks buildable (presubmit compiles both); final notes (summary,
findings, gap) before the end; submit again only for a verified same-bits gain.

Amendment (~16.5 agent-h): s0041 (KVPF) and s0042 (SPF) gave track B 1700 / 3064; s0043 track A 1787 / 3114. The
K/V-prefetch null result at ~13.5 h was a stale library (tools/abs.sh now deletes old libraries). Everything else
tried since (track A placements of the prefetches, KVPF one phase earlier, L2PF, HSK_L2PFW, rolled rescoring, RTASK,
RQ, adaptive head bounds) lost. The run stops adding engine changes here; notes are final.

Amendment (~19.5 agent-h): s0044-s0046 (ETAIL 0, HSK_XG 0, FASTRS_W 1 -- all re-sweeps) took track B to 1692.5 / 3017
(1.298x / 1.135x SGLang). Everything structural tried since lost to the register allocation (journal). Remaining
time: the notes, both tracks buildable; submit again only for a verified same-bits gain.

Amendment (~22.3 agent-h): track B s0047 1690.7 / (s0046) 3017; track A s0048-s0049 crossed the B=8 line: HTAIL
(the exact head's last rowtiles split over the warps, track A only) together with NORM_PIPE1, then KVPF in the
B >= 3 instance (it had been +17 in track A's earlier allocation): 1774.3 / 3074 (1.238x / 1.114x). Both tracks now
clear 1.10x SGLang at both batch sizes. Re-sweeps on both final configurations find nothing better.

Final (~23.4 agent-h): track A s0050 3061 at B=8 (HTAIL tail rowtiles in K halves); nothing further. Both tracks
clear 1.10x SGLang at both batch sizes.

# Plan for the extension (written ~24.1 agent-h; 36 h total)
State: track B s0051 1686.6 / (s0046) 3017; track A s0049 1774.3 / s0050 3061.
Tried first (all measured negative, journal): attention out of line (the call inside the layer loop costs 40-107 us;
the "phases out of line" plan is dead), expert speculation (reuse between steps: B=1 0.28, B=8 0.56), sketch tail
split (STAIL), per-phase prefetch depths (2 is right everywhere).
Remaining options, ranked: (1) check the B=8 choices on more prompt sets (cheap, informs the notes); (2) re-sweeps
after any change that lands; (3) small same-bits ideas in the out-of-line head only (they cannot disturb the layers).
No large rewrites: every structural change so far lost to the register allocation of the main kernel.
