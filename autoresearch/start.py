#!/usr/bin/env python3
"""Start a local autoresearch run: a run directory with a workspace for the agent, the skill, the prompt and `mk`.

    python autoresearch/start.py RUN_DIR --model Qwen/Qwen3-8B --card h100 --hours 12 [--track A|B] [--engine DIR]

--engine is the engine the agent starts from: a directory with engine/ (default: starter/, Hugging Face behind the
engine contract; or one of engines/<model>_<card>/<name> to continue from a zoo engine). Before the first submission the
model's reference must exist on this machine: `bash scripts/eval_selftest.sh MODEL` makes it and checks the evaluator
on the starter. Then give RUN_DIR/PROMPT.md to your agent with RUN_DIR/workspace as its working directory, for example

    cd RUN_DIR/workspace && claude -p "$(cat ../PROMPT.md)" --dangerously-skip-permissions

The agent calls RUN_DIR/bin/mk (on its PATH through the prompt) to submit and to see the board.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

ZOO = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run")
ap.add_argument("--model", required=True, help="Hugging Face id")
ap.add_argument("--card", required=True, help="h100, b200, ...: the name used in eval/baselines_<model>_<card>.json")
ap.add_argument("--hours", type=float, default=12.0)
ap.add_argument("--track", choices=["A", "B"], default="A")
ap.add_argument("--engine", default=str(ZOO / "starter"))
ap.add_argument("--eval", choices=["kernel", "online"], default="kernel")
args = ap.parse_args()

run = Path(args.run).resolve()
if run.exists():
    raise SystemExit(f"{run} exists")
ws = run / "workspace"
shutil.copytree(Path(args.engine) / "engine", ws / "engine", ignore=shutil.ignore_patterns("__pycache__", "*.so", "build"))
(ws / "notes").mkdir()
shutil.copytree(ZOO / "autoresearch" / "skills", ws / ".claude" / "skills")
(run / "submissions").mkdir()
(run / "bin").mkdir()
mk = run / "bin" / "mk"
mk.write_text(f'#!/bin/sh\nMK_RUN="{run}" exec python3 "{ZOO}/autoresearch/mk.py" "$@"\n')
mk.chmod(0o755)
short = args.model.split("/")[-1].lower()
base = ZOO / "eval" / f"baselines_{short}_{args.card}.json"
refs = []
if base.exists():
    for name, s in json.loads(base.read_text()).get("systems", {}).items():
        if s.get("b1"):
            refs.append({"name": f"{name} {s.get('version', '')}".strip()[:80], "value": s["b1"], "note": f"batch 8: {s.get('b8')} us per step"})
if refs:
    fastest = min(r["value"] for r in refs)
    refs.append({"name": "target: 1.10x the fastest general engine", "value": round(fastest / 1.1), "note": "at both batch sizes"})
cfg = {"model": args.model, "card": args.card, "hours": args.hours, "track": args.track, "eval": args.eval, "started_at": time.time(), "reference_lines": refs}
(run / "run.json").write_text(json.dumps(cfg, indent=1) + "\n")
template = (ZOO / "autoresearch" / "prompts" / f"first_{args.track}.md").read_text()
baseline = "; ".join(f"{r['name']}: {r['value']} us per token" for r in refs[:-1]) or "measure the general engines on your card first (scripts/compare.sh) and add them to run.json"
prompt = template.replace("<model>", args.model).replace("<card>", args.card).replace("<hours>", f"{args.hours:g}").replace("<baseline>", baseline).replace("<mk>", str(mk))
(run / "PROMPT.md").write_text(prompt)
print(f"run ready: {run}\n  workspace  {ws}\n  prompt     {run / 'PROMPT.md'}\n  tool       {mk}\n  references {len(refs)} line(s) from {base.name if base.exists() else 'none (no baselines file for this model and card)'}")
