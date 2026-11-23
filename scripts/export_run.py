#!/usr/bin/env python
"""A run directory's record in the form of runs/: progress.md (every submission: what was tried, why, what it
measured), src/sNNNN.tgz (the source of each submission) and notes/ (the agent's own notes).

    python scripts/export_run.py RUN_DIR [--out OUT_DIR] [--title "Inkling-2-Mini on Apple M5 Max"]

Reads RUN_DIR/run.json, ledger.jsonl, submissions/sNNNN/src.tgz and workspace/notes; writes into OUT_DIR (default:
RUN_DIR itself, next to the raw files). The agent transcripts and the workspace are not part of the record.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    out = Path(args.out) if args.out else run
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((run / "run.json").read_text())
    rows = [json.loads(line) for line in (run / "ledger.jsonl").read_text().splitlines() if line.strip()]
    metric = rows[0]["metric"] if rows else "?"
    lower = rows[0].get("lower_is_better", True) if rows else True
    valid = sum(r["status"] == "valid" for r in rows)
    title = args.title or f"{meta['model'].split('/')[-1]} on {meta['card']}"
    lines = [f"# {run.name}: {title}", "",
             f"Score: `{metric}`" + (" (lower is better)" if lower else "") + f". Budget {float(meta['hours']):.1f} agent-hours. "
             f"Track {meta.get('track', '?')}. {len(rows)} submissions, {valid} valid.", ""]
    if meta.get("continues"):
        lines += [f"Continues {meta['continues']}.", ""]
    if meta.get("reference_lines"):
        lines += ["Reference lines of this run:", ""]
        lines += [f"- {r['name']}: {r['value']}" + (f" ({r['note']})" if r.get("note") else "") for r in meta["reference_lines"]]
        lines += [""]
    lines += ["The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.", "", "## Submissions", ""]
    (out / "src").mkdir(exist_ok=True)
    for i, r in enumerate(rows, 1):
        value = f"{metric} = {r['value']:.0f}" if isinstance(r.get("value"), (int, float)) else "no score"
        head = f"- status: **{r['status']}**, {value}, at {r['hours']:.2f} agent-hours"
        if r.get("parent"):
            head += f", parent {r['parent']}"
        if r.get("category"):
            head += f", category {r['category']}"
        if r.get("quick"):
            head += ", quick evaluation"
        lines += [f"### #{i} {r['sid']}: {r['title']}", "", head, f"- hypothesis: {r['hypothesis']}", f"- evaluator: {r['message']}", ""]
        src = run / "submissions" / r["sid"] / "src.tgz"
        if src.exists():
            shutil.copyfile(src, out / "src" / f"{r['sid']}.tgz")
    (out / "progress.md").write_text("\n".join(lines))
    notes = run / "workspace" / "notes"
    if notes.is_dir() and any(notes.iterdir()):
        shutil.copytree(notes, out / "notes", dirs_exist_ok=True)
    print(f"{out}: {len(rows)} submissions, {len(list((out / 'src').glob('*.tgz')))} sources, notes: {(out / 'notes').is_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
