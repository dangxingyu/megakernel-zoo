#!/usr/bin/env python
"""Evaluate an engine online: requests arrive and leave, as in a production trace.

    python eval/online_eval.py --workspace WS --out RESULT.json [--quick] [--rates "2"]

Order: (1) the offline evaluation (eval/kernel_eval.py --quick --no-trace): the engine must still be correct
against float64 step by step; (2) a replay of the Mooncake tool & agent trace through the engine's online
interface (eval/_online_runner.py has the contract), the same requests at the same arrival times that the
general engines were measured on (scripts/e16_replay.sh); (3) invariants of an engine that owns its own
scheduling and memory.

Gates (all must pass for `valid`):
  offline            every gate of the offline evaluation
  online_api         submit / poll / idle / kv_pages / drop_cache exist
  liveness           every request of the replay finished with exactly the tokens it asked for
  schedule           a request's tokens under load equal its tokens alone on an idle engine with an empty cache
                     (probe requests; the engine's batching, prefix reuse and eviction must not change a result)
  online_vs_step     the offline reference prompts through submit/poll give the step-by-step greedy tokens,
                     alone and all together
  kv_conservation    idle engine after drop_cache: free pages == total pages, at the start, after every replay
                     and at the end (nothing leaks, nothing is counted twice)
The two token gates allow a sequence to differ only rarely (near-ties): at least 75% of sequences identical and a
mean matching prefix of at least 0.9; `identical` counts are recorded, aim for all.

Headline: the share of the replay's requests that meet the service levels (TTFT <= 5 s and mean time between
tokens <= 0.1 s) at the standard rate (nominal 2 req/s, 120 s): higher is better. TTFT, time between tokens and
the longest stall are reported next to the general engines' numbers on the same requests.
Timing: in process (no HTTP, no detokenization); a token's time is when poll() first returned it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

_SETUP = {}
_setup_file = Path(os.environ.get("MK_ROOT", "/tmp/mk")) / "eval_env.json"
if _setup_file.exists():
    _SETUP = json.loads(_setup_file.read_text()).get("private", {})
    if _SETUP.get("python") and os.path.realpath(sys.executable) != os.path.realpath(_SETUP["python"]) and not os.environ.get("MK_EVAL_REEXEC"):
        os.environ["MK_EVAL_REEXEC"] = "1"
        os.execv(_SETUP["python"], [_SETUP["python"], *sys.argv])

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prompts  # noqa: E402

TRACE_RPS = 6.67  # the tool & agent trace's own rate (scripts/e16_replay.sh uses the same constant)
SLO_TTFT_S, SLO_TBT_S = 5.0, 0.1


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else float("nan")


def plan_replay(trace: Path, pool: list[int], rate: float, duration: float, max_input: int, max_output: int, base_rid: int) -> list[dict]:
    """The same selection and synthesis as eval/client.py replay: identical requests and arrival times."""
    scale = rate / TRACE_RPS
    rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    t_first, out = rows[0]["timestamp"], []
    for row in rows:
        at = (row["timestamp"] - t_first) / 1e3 / scale
        if at > duration:
            break
        n_in = min(row["input_length"], max_input)
        out.append({"rid": base_rid + len(out), "at": at, "ids": prompts.from_block_hashes(pool, row["hash_ids"], n_in),
                    "n_out": max(1, min(row["output_length"], max_output))})
    return out


def summarize(replay: dict, duration: float) -> dict:
    records = replay["records"]
    done = [r for r in records if r["finished"] and r["token_times"]]
    ttft = [r["token_times"][0] - r["t_submit"] for r in done]
    gaps = [(b - a) * 1e3 for r in done for a, b in zip(r["token_times"], r["token_times"][1:])]
    tbt = [(r["token_times"][-1] - r["token_times"][0]) / max(1, len(r["token_times"]) - 1) for r in done if len(r["token_times"]) > 1]
    within = sum(1 for r in done if r["token_times"][0] - r["t_submit"] <= SLO_TTFT_S and
                 (len(r["token_times"]) < 2 or (r["token_times"][-1] - r["token_times"][0]) / (len(r["token_times"]) - 1) <= SLO_TBT_S))
    wrong_length = sum(1 for r in done if len(r["tokens"]) != r["n_out"])
    late = max((r["t_submit"] - r["planned_at"] for r in records if r["t_submit"] is not None), default=0.0)

    def pcts(values: list[float], scale: float = 1.0) -> dict | None:
        return {"p50": pct(values, 0.5) * scale, "p90": pct(values, 0.9) * scale, "p99": pct(values, 0.99) * scale,
                "max": max(values) * scale} if values else None

    return {"planned": len(records), "offered_req_s": len(records) / duration, "finished": len(done),
            "unfinished": len(records) - len(done), "wrong_length": wrong_length,
            "input_tokens_mean": float(np.mean([r["n_in"] for r in records])), "output_tokens_mean": float(np.mean([r["n_out"] for r in records])),
            "output_tok_s": sum(len(r["tokens"]) for r in done) / replay["wall_s"], "completed_req_s": len(done) / replay["wall_s"],
            "ttft_ms": pcts(ttft, 1e3), "gap_ms": pcts(gaps), "tbt_ms": pcts(tbt, 1e3),
            "slo": {"ttft_s": SLO_TTFT_S, "tbt_s": SLO_TBT_S, "within": within, "share_of_planned": within / max(1, len(records))},
            "submit_max_ms": max((r["submit_seconds"] or 0.0) for r in records) * 1e3, "driver_lateness_max_ms": late * 1e3,
            "poll_max_ms": replay["poll_max_ms"], "poll_calls": replay["poll_calls"], "wall_s": replay["wall_s"],
            "idle_after": replay["idle_after"], "pages_after": replay["pages_after"], "pages_after_drop": replay["pages_after_drop"],
            "engine_stats": replay.get("engine_stats"),
            "timing": "in process, open loop at the trace's arrival times; a token's time is when poll() returned it"}


def same_tokens(pairs: list[tuple[list[int], list[int]]]) -> dict:
    """How well pairs of token sequences agree: identical count and mean matching prefix."""
    if not pairs:
        return {"pass": False, "note": "nothing to compare"}
    prefix, identical = [], 0
    for a, b in pairs:
        n = min(len(a), len(b))
        k = next((i for i in range(n) if a[i] != b[i]), n)
        full = len(a) == len(b) and k == n
        identical += full
        prefix.append(1.0 if full else k / max(1, max(len(a), len(b))))
    mean = float(np.mean(prefix))
    return {"pass": identical >= 0.75 * len(pairs) and mean >= 0.9, "identical": identical, "of": len(pairs), "mean_matching_prefix": mean}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quick", action="store_true", help="40 s of the replay instead of 120 s, 6 probes")
    parser.add_argument("--rates", default="2", help="nominal request rates in req/s; the first is the headline one")
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--model-dir", default=os.environ.get("MK_MODEL_DIR"))
    parser.add_argument("--ref", default=os.environ.get("MK_REF_DIR") or _SETUP.get("ref"))
    parser.add_argument("--trace", default=os.environ.get("MK_TRACE") or _SETUP.get("trace"))
    parser.add_argument("--baselines", default=os.environ.get("MK_ONLINE_BASELINES") or _SETUP.get("online_baselines"))
    parser.add_argument("--python", default=os.environ.get("MK_EVAL_PY") or _SETUP.get("python") or sys.executable)
    parser.add_argument("--scratch", default=os.environ.get("MK_EVAL_SCRATCH", "/dev/shm"))
    args = parser.parse_args()
    t_begin = time.time()
    workspace = str(Path(args.workspace).resolve())
    scratch = Path(args.scratch) / f"mk-online-{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=True)
    duration = 40.0 if args.quick else args.duration_s
    result: dict = {"status": "error", "mode": "online", "quick": args.quick, "gates": {}, "online": {}, "milestones_passed": []}

    def finish(message: str) -> int:
        result["seconds"] = round(time.time() - t_begin, 1)
        result["message"] = message
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
        subprocess.run(["rm", "-rf", str(scratch)])
        print("MK_EVAL " + json.dumps({k: result.get(k) for k in ("status", "headline", "milestones_passed", "seconds", "message")}))
        return 0

    # ---- (1) offline: the engine must still compute the model
    offline_out = scratch / "offline.json"
    done = subprocess.run([args.python, str(HERE / "kernel_eval.py"), "--workspace", workspace, "--out", str(offline_out), "--quick", "--no-trace"],
                          capture_output=True, text=True, timeout=3600)
    offline = json.loads(offline_out.read_text()) if offline_out.exists() else {"status": "error", "message": (done.stdout + done.stderr)[-1500:]}
    result["offline"] = {k: offline.get(k) for k in ("status", "message", "gates", "perf", "build", "milestones_passed", "seconds")}
    result["gates"]["offline"] = {"pass": offline.get("status") == "valid", "status": offline.get("status")}
    if offline.get("status") == "build_failed":
        result["status"] = "build_failed"
        return finish("offline evaluation: " + str(offline.get("message")))
    if offline.get("status") != "valid":
        result["status"] = "incorrect" if offline.get("status") == "incorrect" else "error"
        return finish("offline evaluation: " + str(offline.get("message")))
    result["milestones_passed"] = list(offline.get("milestones_passed") or [])

    # ---- (2) the plan: identical requests to the baseline replays
    tok = prompts.load(args.model_dir)
    pool = prompts.word_pool(tok)
    rates = [float(r) for r in args.rates.split()]
    replays = {f"r{rate:g}": plan_replay(Path(args.trace), pool, rate, duration, args.max_input_tokens, args.max_output_tokens, 1_000_000 * (i + 1))
               for i, rate in enumerate(rates)}
    headline_name = f"r{rates[0]:g}"
    rng = random.Random(20261110)
    candidates = [r["rid"] for r in replays[headline_name] if len(r["ids"]) <= 8192]
    probe = sorted(rng.sample(candidates, min(len(candidates), 6 if args.quick else 12)))
    meta = json.loads((Path(args.ref) / "meta.json").read_text())
    ids = np.load(Path(args.ref) / "ids.npy")
    P, T = meta["prompt_tokens"], meta["check_tokens"]
    plan = {"replays": replays, "probe": probe, "reference_prompts": [ids[r, :P].tolist() for r in range(ids.shape[0])],
            "reference_new_tokens": T - P}
    (scratch / "plan.json").write_text(json.dumps(plan))
    done = subprocess.run([args.python, str(HERE / "_online_runner.py"), "--workspace", workspace, "--model-dir", args.model_dir,
                           "--plan", str(scratch / "plan.json"), "--out", str(scratch / "online.json")],
                          capture_output=True, text=True, timeout=3600, cwd=workspace)
    if done.returncode != 0 or not (scratch / "online.json").exists():
        result["gates"]["online_run"] = {"pass": False, "rc": done.returncode, "tail": (done.stdout + done.stderr)[-2500:]}
        result["status"] = "incorrect"
        return finish(f"the engine failed in the online run (rc {done.returncode})")
    report = json.loads((scratch / "online.json").read_text())
    gates = result["gates"]
    gates["online_api"] = {"pass": not report.get("missing_methods"), "missing": report.get("missing_methods")}
    if report.get("missing_methods"):
        result["status"] = "incorrect"
        return finish("online interface missing: " + ", ".join(report["missing_methods"]))

    # ---- (3) numbers and invariants
    for name, replay in report["replays"].items():
        result["online"][name] = summarize(replay, duration)
    head = result["online"][headline_name]
    gates["liveness"] = {"pass": all(o["unfinished"] == 0 and o["wrong_length"] == 0 and o["idle_after"] for o in result["online"].values()),
                         "unfinished": {k: o["unfinished"] for k, o in result["online"].items()},
                         "wrong_length": {k: o["wrong_length"] for k, o in result["online"].items()}}
    loaded = {r["rid"]: r["tokens"] for r in report["replays"][headline_name]["records"]}
    gates["schedule"] = {**same_tokens([(loaded[int(rid)], rec["tokens"]) for rid, rec in report["alone"].items()]),
                         "note": "probe requests: tokens in the replay against tokens alone on an idle engine with an empty cache"}
    step_tokens = offline.get("free_run_tokens")
    if step_tokens:
        alone = same_tokens(list(zip(report.get("reference_alone", []), step_tokens)))
        together = same_tokens(list(zip(report.get("reference_together", []), step_tokens)))
        gates["online_vs_step"] = {"pass": alone["pass"] and together["pass"], "alone": alone, "together": together,
                                   "note": "the offline reference prompts through submit/poll against the step-by-step greedy tokens"}
    else:
        gates["online_vs_step"] = {"pass": False, "note": "the offline evaluation returned no step-by-step tokens"}
    snapshots = [report["pages_at_start"], report["pages_at_end"]] + [r["pages_after_drop"] for r in report["replays"].values()]
    gates["kv_conservation"] = {"pass": all(free == total and total > 0 for free, total in snapshots), "free_total": snapshots,
                                "note": "idle engine after drop_cache: free pages must equal total pages"}
    valid = all(g.get("pass") for g in gates.values())
    result["status"] = "valid" if valid else "incorrect"
    if args.baselines and Path(args.baselines).exists():
        lines = json.loads(Path(args.baselines).read_text())
        result["baselines"] = lines
        compare = {}
        for system, per_rate in lines.get("systems", {}).items():
            base = per_rate.get(headline_name)
            if base:
                compare[system] = {"slo_share": [head["slo"]["share_of_planned"], base["slo_share"]],
                                   "ttft_p50_ms": [head["ttft_ms"] and head["ttft_ms"]["p50"], base["ttft_p50_ms"]],
                                   "tbt_p99_ms": [head["tbt_ms"] and head["tbt_ms"]["p99"], base["tbt_p99_ms"]],
                                   "gap_max_ms": [head["gap_ms"] and head["gap_ms"]["max"], base["gap_max_ms"]]}
        result["compare_engine_then_baseline"] = compare
    if not valid:
        return finish("incorrect: failed " + ", ".join(k for k, g in gates.items() if not g.get("pass")))
    result["milestones_passed"].append("online")
    result["headline"] = {"metric": f"slo_share_{headline_name}", "value": head["slo"]["share_of_planned"], "lower_is_better": False}
    best = max((v["slo_share"][1] for v in result.get("compare_engine_then_baseline", {}).values()), default=None)
    if best is not None and head["slo"]["share_of_planned"] > best:
        result["milestones_passed"].append("online_beats_engines")
    return finish(f"valid online: {head['slo']['share_of_planned']:.3f} of {head['planned']} requests within the service levels at "
                  f"{head['offered_req_s']:.2f} req/s; TTFT p50 {head['ttft_ms']['p50']:.0f} ms, time between tokens p99 {head['tbt_ms']['p99']:.0f} ms")


if __name__ == "__main__":
    raise SystemExit(main())
