#!/usr/bin/env python3
"""Start a local autoresearch run: a run directory with a workspace for the agent, the skill, the prompt and `mk`.

    python autoresearch/start.py RUN_DIR --model Qwen/Qwen3-8B --card h100 --hours 12 [--track A|B|C] [--engine DIR]
    python autoresearch/start.py RUN_DIR --model thinkingmachines/Inkling-2-Mini --card m5max --track C --hours 12

--engine is the engine the agent starts from: a directory with engine/ (default: starter/, Hugging Face behind the
engine contract; or one of engines/<model>_<card>/<name> to continue from a zoo engine). Before the first submission the
model's reference must exist on this machine: `bash scripts/eval_selftest.sh MODEL` makes it and checks the evaluator
on the starter. Then give RUN_DIR/PROMPT.md to your agent with RUN_DIR/workspace as its working directory, for example

    cd RUN_DIR/workspace && claude -p "$(cat ../PROMPT.md)" --dangerously-skip-permissions

The agent calls RUN_DIR/bin/mk (on its PATH through the prompt) to submit and to see the board.

Track C (exact draft-and-verify): the reference lines come from eval/baselines_<model>_<card>_c.json and RUN_DIR/bin/mk
runs the evaluator with MK_TRACK=c. An Apple-silicon card (`apple` in eval/modelinfo.py CARDS, e.g. m5max) starts from
starter_mac/ (mlx-lm behind the contract), gets the Mac prompt (autoresearch/prompts/first_<track>_mac.md) and the Mac
skill (skills/megakernel/SKILL_mac.md as the workspace's SKILL.md), and RUN_DIR/bin/mk runs under .venv-mac; prepare the
Mac once with `bash scripts/eval_setup_mac.sh`.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ZOO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ZOO / "eval"))
import modelinfo  # noqa: E402  (standard library only)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run")
ap.add_argument("--model", required=True, help="Hugging Face id")
ap.add_argument("--card", required=True, help="h100, b200, m5max, ...: the name used in eval/baselines_<model>_<card>.json")
ap.add_argument("--hours", type=float, default=12.0)
ap.add_argument("--track", choices=["A", "B", "C"], type=str.upper, default="A")
ap.add_argument("--engine", default=None, help="default: starter/ (starter_mac/ on an Apple-silicon card)")
ap.add_argument("--eval", choices=["kernel", "online"], default="kernel")
ap.add_argument("--model-dir", default=None, help="the local checkpoint, handed to the evaluator as MK_MODEL_DIR when the "
                                                  "environment has none (default on a Mac: ~/models/<the id's last part>)")
ap.add_argument("--ref-dir", default=None, help="the float64 reference, handed to the evaluator as MK_REF_DIR when the environment has none")
args = ap.parse_args()
apple = bool(modelinfo.CARDS.get(args.card.lower(), {}).get("apple"))
if args.engine is None:
    args.engine = str(ZOO / ("starter_mac" if apple else "starter"))
if apple and args.model_dir is None:
    args.model_dir = str(Path.home() / "models" / args.model.split("/")[-1])
suffix = "_mac" if apple else ""
template_path = ZOO / "autoresearch" / "prompts" / f"first_{args.track}{suffix}.md"
if not template_path.exists():
    raise SystemExit(f"no prompt for track {args.track} on {args.card}: {template_path} does not exist")

run = Path(args.run).resolve()
if run.exists():
    raise SystemExit(f"{run} exists")
ws = run / "workspace"
shutil.copytree(Path(args.engine) / "engine", ws / "engine", ignore=shutil.ignore_patterns("__pycache__", "*.so", "build"))
(ws / "notes").mkdir()
shutil.copytree(ZOO / "autoresearch" / "skills", ws / ".claude" / "skills", ignore=shutil.ignore_patterns("SKILL_mac.md"))
if apple:  # the Mac's skill replaces the CUDA one (same name, so the agent loads `megakernel` either way)
    shutil.copy(ZOO / "autoresearch" / "skills" / "megakernel" / "SKILL_mac.md", ws / ".claude" / "skills" / "megakernel" / "SKILL.md")
(run / "submissions").mkdir()
(run / "bin").mkdir()
mk = run / "bin" / "mk"
# track C: the evaluator's rules for exact draft-and-verify (MK_TRACK=c; mk.py also passes --track c). A Mac: the
# evaluator runs under the repository's .venv-mac (MLX), whatever python3 is first on the PATH.
driver = str(ZOO / ".venv-mac" / "bin" / "python") if apple else "python3"
track_env = " MK_TRACK=c" if args.track == "C" else ""
mk.write_text(f'#!/bin/sh\nMK_RUN="{run}"{track_env} exec {driver} "{ZOO}/autoresearch/mk.py" "$@"\n')
mk.chmod(0o755)
short = args.model.split("/")[-1].lower()
base = ZOO / "eval" / f"baselines_{short}_{args.card}{'_c' if args.track == 'C' else ''}.json"
refs = []
if base.exists():
    for name, s in json.loads(base.read_text()).get("systems", {}).items():
        if s.get("b1"):
            note = f"batch 8: {s.get('b8')} us per step" if s.get("b8") else ""
            if s.get("b1_natural"):
                note += ("; " if note else "") + f"natural text at batch 1: {s['b1_natural']} us per token"
            refs.append({"name": f"{name} {s.get('version', '')}".strip()[:80], "value": s["b1"], "note": note})
if refs:
    fastest = min(r["value"] for r in refs)
    refs.append({"name": "target: 1.10x the fastest general engine", "value": round(fastest / 1.1),
                 "note": "at batch 1" if apple else "at both batch sizes"})
cfg = {"model": args.model, "card": args.card, "hours": args.hours, "track": args.track, "eval": args.eval, "started_at": time.time(), "reference_lines": refs}
if args.model_dir:
    cfg["model_dir"] = args.model_dir
if args.ref_dir:
    cfg["ref_dir"] = args.ref_dir
if apple:
    cfg["eval_timeout_s"] = 3 * 3600  # an MLX starter at tens of ms per token: check + time + HTTP take a while
(run / "run.json").write_text(json.dumps(cfg, indent=1) + "\n")
template = template_path.read_text()
baseline = "; ".join(f"{r['name']}: {r['value']} us per token" + (f" ({r['note']})" if apple and r["note"] else "") for r in refs[:-1]) \
    or "measure the general engines on your card first (scripts/compare.sh; on a Mac baselines/mac/bench.sh) and add them to run.json"
prompt = template.replace("<model>", args.model).replace("<card>", args.card).replace("<hours>", f"{args.hours:g}").replace("<baseline>", baseline).replace("<mk>", str(mk))
(run / "PROMPT.md").write_text(prompt)
print(f"run ready: {run}\n  workspace  {ws}\n  prompt     {run / 'PROMPT.md'}\n  tool       {mk}\n  references {len(refs)} line(s) from {base.name if base.exists() else 'none (no baselines file for this model and card)'}")
