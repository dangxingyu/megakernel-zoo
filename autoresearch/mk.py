#!/usr/bin/env python3
"""mk: the agent's tool in a local autoresearch run (agent and GPU on the same machine).

    mk submit --title T --hypothesis H [--category C] [--quick]   evaluate engine/ as it is now; the only thing that counts
    mk board                                                     every submission, the best so far, reference lines, time left
    mk note "text"                                               a line on the run's timeline

`mk submit` copies the workspace's engine/ to a sealed directory, runs the run's evaluator on the copy and appends the
result to RUN/ledger.jsonl; the source and the full result are kept under RUN/submissions/<sid>/. The run directory is
$MK_RUN (the wrapper RUN/bin/mk sets it). Standard library only.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

RUN = Path(os.environ.get("MK_RUN") or ".").resolve()
ZOO = Path(__file__).resolve().parents[1]


def cfg() -> dict:
    return json.loads((RUN / "run.json").read_text())


def rows() -> list[dict]:
    path = RUN / "ledger.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def hours_used(c: dict) -> float:
    return (time.time() - c["started_at"]) / 3600


def submit(args) -> int:
    c = cfg()
    if hours_used(c) >= c["hours"]:
        print(f"the run's {c['hours']:g} hours are used up; nothing more is evaluated")
        return 1
    sid = f"s{len(rows()) + 1:04d}"
    sub = RUN / "submissions" / sid
    sealed = sub / "sealed"
    shutil.copytree(RUN / "workspace" / "engine", sealed / "engine", ignore=shutil.ignore_patterns("__pycache__", "*.so", "build"))
    with tarfile.open(sub / "src.tgz", "w:gz") as tar:
        tar.add(sealed / "engine", arcname="engine")
    evaluator = ZOO / "eval" / ("online_eval.py" if c.get("eval") == "online" else "kernel_eval.py")
    cmd = [sys.executable, str(evaluator), "--workspace", str(sealed), "--out", str(sub / "result.json")] + (["--quick"] if args.quick else [])
    print(f"{sid}: evaluating (this takes minutes; log in {sub / 'eval.log'})", flush=True)
    t0 = time.time()
    with open(sub / "eval.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ZOO, timeout=c.get("eval_timeout_s", 5400)).returncode
    result = json.loads((sub / "result.json").read_text()) if (sub / "result.json").exists() else {"status": "error", "message": f"the evaluator exited with {rc}; see eval.log"}
    head = result.get("headline") or {}
    row = {"sid": sid, "ts": time.time(), "hours": round(hours_used(c), 3), "title": args.title, "hypothesis": args.hypothesis, "category": args.category,
           "quick": bool(args.quick), "status": result.get("status"), "metric": head.get("metric"), "value": head.get("value"),
           "lower_is_better": head.get("lower_is_better", True), "message": result.get("message"), "eval_seconds": round(time.time() - t0, 1)}
    with open(RUN / "ledger.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    shutil.rmtree(sealed, ignore_errors=True)
    print(f"{sid}: {row['status']}" + (f", {row['metric']} = {row['value']:.4g}" if row["value"] is not None else "") + f"\n{row['message']}")
    gates = {k: v.get("pass") for k, v in (result.get("gates") or {}).items()}
    if gates:
        print("gates: " + ", ".join(f"{k} {'ok' if v else 'FAILED'}" for k, v in gates.items()))
    return 0


def board(_args) -> int:
    c, rs = cfg(), rows()
    left = c["hours"] - hours_used(c)
    print(f"{c['model']} on {c['card']}: {hours_used(c):.2f} of {c['hours']:g} hours used, {max(left, 0):.2f} left; {len(rs)} submissions")
    for ref in c.get("reference_lines", []):
        print(f"  reference  {ref['value']:>10.4g}  {ref['name']}" + (f"  ({ref['note']})" if ref.get("note") else ""))
    ranked = [r for r in rs if r["status"] == "valid" and r["value"] is not None and not r["quick"]]
    if ranked:
        best = (min if ranked[0]["lower_is_better"] else max)(ranked, key=lambda r: r["value"])
        print(f"  best       {best['value']:>10.4g}  {best['sid']}  {best['title']}")
    for r in rs:
        value = f"{r['value']:.4g}" if r["value"] is not None else "-"
        print(f"  {r['sid']}  {r['hours']:6.2f} h  {r['status'] or '?':9s} {value:>10s}  {'quick ' if r['quick'] else ''}{r['title']}")
    return 0


def note(args) -> int:
    with open(RUN / "notes.jsonl", "a") as f:
        f.write(json.dumps({"ts": time.time(), "hours": round(hours_used(cfg()), 3), "text": args.text}) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--title", required=True)
    s.add_argument("--hypothesis", required=True, help="what you changed, why it should help, which number should move by how much")
    s.add_argument("--category", default="other")
    s.add_argument("--quick", action="store_true", help="a shorter evaluation, logged but never ranked")
    s.set_defaults(fn=submit)
    sub.add_parser("board").set_defaults(fn=board)
    n = sub.add_parser("note")
    n.add_argument("text")
    n.set_defaults(fn=note)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
