#!/usr/bin/env python3
"""Turn a dump of mkprof.cuh marks into a table: where one decode step's time goes inside the persistent kernel.

    python tools/mkprof/mkprof.py PROF.npy [--tags TAGS.json] [--bytes BYTES.json] [--card h100|b200|GBPS] [--json OUT]

PROF.npy: int64 array [blocks, cap, 2] as written by the kernel (rows of zeros = unused records).
TAGS.json: {"1": "qkv", "2": "attn", ...} (tag number -> phase name); tags 65534 (step) and 65535 (wait) are built in.
BYTES.json: {"qkv": 123456789, ...} bytes one occurrence of the phase must read (all blocks together): gives the
bandwidth the phase reached over its span and its share of the card's peak.

For every phase tag, over all its occurrences (the k-th occurrence in every block is taken as the same instance,
e.g. the same layer of the same step):
  span      first block in -> last block out of the phase (where the grid as a whole spends the time)
  work      per block: its mark -> its next mark that is not a wait (its own work in the phase)
  wait      per block: time in a WAIT mark's interval directly after the phase (barrier, flag)
  imb       max over blocks / median over blocks of `work` (1.0 = perfectly balanced)
Durations use each SM's cycle counter converted to ns with the clock rate fitted against the global timer, so they
are not limited by the global timer's update interval; the global timer orders blocks against each other.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict

import numpy as np

STEP, WAIT = 0xFFFE, 0xFFFF
CARDS = {"h100": 3350.0, "b200": 8000.0}


def load(path):
    a = np.load(path)
    a = a.reshape(a.shape[0], -1, 2).astype(np.uint64)
    blocks = []
    for b in range(a.shape[0]):
        rows = a[b]
        n = int(np.argmax(rows[:, 0] == 0)) if (rows[:, 0] == 0).any() else rows.shape[0]
        g = rows[:n, 0].astype(np.float64)
        cyc = (rows[:n, 1] >> np.uint64(16)).astype(np.float64)
        tag = (rows[:n, 1] & np.uint64(0xFFFF)).astype(np.int64)
        blocks.append((g, cyc, tag))
    return blocks


def ns_per_cycle(blocks):
    """Fit the global timer against the cycle counter on the longest block (the SM clock may differ from nominal)."""
    g, c, _ = max(blocks, key=lambda x: len(x[0]))
    if len(g) < 3 or c[-1] == c[0]:
        return None
    return float(np.polyfit(c - c[0], g - g[0], 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prof")
    ap.add_argument("--tags")
    ap.add_argument("--bytes")
    ap.add_argument("--card", default=None)
    ap.add_argument("--json")
    args = ap.parse_args()
    names = {int(k): v for k, v in json.load(open(args.tags)).items()} if args.tags else {}
    names.setdefault(STEP, "step"); names.setdefault(WAIT, "wait")
    nbytes = json.load(open(args.bytes)) if args.bytes else {}
    peak = CARDS.get(str(args.card).lower()) if args.card and not str(args.card).replace(".", "").isdigit() else (float(args.card) if args.card else None)
    blocks = load(args.prof)
    k = ns_per_cycle(blocks)
    used = [len(b[0]) for b in blocks]
    print(f"blocks {len(blocks)}, records per block min/median/max {min(used)}/{int(st.median(used))}/{max(used)}"
          + (f", SM clock {1 / k:.3f} GHz (fitted)" if k else ", no clock fit (too few records)"))
    if max(used) == 0:
        return
    # occurrences: inst[tag][j] = list over blocks of (start_g, work_ns, wait_ns, end_g)
    inst = defaultdict(lambda: defaultdict(list))
    step_ns = []
    for g, c, t in blocks:
        seen = defaultdict(int)
        for i in range(len(t) - 1):
            tag = int(t[i])
            if tag == WAIT:
                continue
            dur = (c[i + 1] - c[i]) * k if k else (g[i + 1] - g[i])
            wait = 0.0
            j = i + 1
            while j < len(t) - 1 and int(t[j]) == WAIT:
                wait += (c[j + 1] - c[j]) * k if k else (g[j + 1] - g[j])
                j += 1
            if tag == STEP:
                continue
            inst[tag][seen[tag]].append((g[i], dur, wait, g[j]))
            seen[tag] += 1
        steps = [i for i in range(len(t)) if int(t[i]) == STEP]
        step_ns += [g[b] - g[a] for a, b in zip(steps, steps[1:])]
    t0 = blocks[0][2]  # steps that were recorded whole: a STEP mark that is not the block's last record
    nsteps = max(1, sum(1 for i, x in enumerate(t0) if int(x) == STEP and i < len(t0) - 1
                        and any(int(y) != STEP for y in t0[i + 1:])))
    step = st.median(step_ns) if step_ns else None
    rows = []
    for tag, occ in inst.items():
        spans, works, waits, imbs = [], [], [], []
        for j, lst in occ.items():
            if len(lst) < 1:
                continue
            spans.append(max(x[3] for x in lst) - min(x[0] for x in lst))
            w = [x[1] for x in lst]
            works.append(st.median(w)); imbs.append(max(w) / max(st.median(w), 1e-9))
            waits.append(st.mean(x[2] for x in lst))
        name = names.get(tag, str(tag))
        row = {"phase": name, "tag": tag, "occurrences": len(spans), "per_step": len(spans) / nsteps,
               "span_us": st.mean(spans) / 1e3, "work_us_median_block": st.mean(works) / 1e3,
               "wait_us_mean_block": st.mean(waits) / 1e3, "imbalance": st.mean(imbs),
               "span_us_per_step": sum(spans) / nsteps / 1e3}
        if name in nbytes:
            gbs = nbytes[name] / (row["span_us"] * 1e3)
            row["gb_s"] = gbs
            if peak:
                row["pct_peak"] = 100 * gbs / peak
        rows.append(row)
    rows.sort(key=lambda r: -r["span_us_per_step"])
    covered = sum(r["span_us_per_step"] for r in rows)
    if step:
        print(f"steps {nsteps}, step time median {step / 1e3:.1f} us; phase spans cover {covered:.1f} us per step "
              f"({100 * covered / (step / 1e3):.0f}%; overlap between phases counts twice, gaps not at all)")
    hdr = f"{'phase':16s} {'per step':>8s} {'span us':>9s} {'us/step':>9s} {'%step':>6s} {'work us':>8s} {'wait us':>8s} {'imb':>5s} {'GB/s':>7s} {'%peak':>6s}"
    print(hdr)
    for r in rows:
        pct = 100 * r["span_us_per_step"] / (step / 1e3) if step else float("nan")
        print(f"{r['phase'][:16]:16s} {r['per_step']:8.1f} {r['span_us']:9.2f} {r['span_us_per_step']:9.1f} {pct:6.1f} "
              f"{r['work_us_median_block']:8.2f} {r['wait_us_mean_block']:8.2f} {r['imbalance']:5.2f} "
              f"{r.get('gb_s', float('nan')):7.0f} {r.get('pct_peak', float('nan')):6.1f}")
    if args.json:
        json.dump({"step_us": step / 1e3 if step else None, "steps": nsteps, "phases": rows,
                   "ns_per_cycle": k}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
