#!/usr/bin/env python
"""Child process of online_eval.py: drives the engine under test through its online interface. Not called by hand.

The online contract (same class as the offline one, WS/engine/api.py):

    class Engine:
        def __init__(self, model_dir: str, max_batch: int, max_context: int): ...
        def submit(self, rid: int, prompt: list[int], max_new_tokens: int) -> None
            # A request arrives. Must return at once (no model work for this request before returning beyond
            # copying the prompt). Greedy decoding; exactly max_new_tokens tokens are produced (no early stop).
        def poll(self) -> list[tuple[int, list[int], bool]]
            # New tokens since the last poll, per request, in generation order: (rid, tokens, finished).
            # May do work (a host-driven engine can run one step here) but must return within a few steps.
        def idle(self) -> bool           # nothing queued and nothing running
        def kv_pages(self) -> tuple[int, int]      # (free, total) pages of the engine's KV store right now
        def drop_cache(self) -> None     # forget every retained prefix (after it, an idle engine has free == total)
        def stats(self) -> dict          # optional counters (cache hits, evictions, recomputes, queue length ...)

Admission, batching, prefill, KV allocation, prefix reuse, eviction and recomputation are the engine's own
business: the driver only says when a request arrives and reads tokens.

This process produces raw material only: for every request the submit time and the time each token was first
returned by poll(). Verdicts are made by the parent.

Modes:
  replay   requests from a plan file (arrival second, prompt ids, output length), open loop, then a drain.
  alone    the plan's `probe` requests one at a time on an otherwise idle engine (after drop_cache): the tokens a
           request must also produce under load.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


def load_engine(workspace: str, model_dir: str, max_batch: int, max_context: int):
    api = Path(workspace) / "engine" / "api.py"
    sys.path.insert(0, str(api.parent))
    spec = importlib.util.spec_from_file_location("engine_api", api)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Engine(model_dir, max_batch, max_context)


def drive(engine, requests: list[dict], open_loop: bool, drain_s: float, poll_sleep: float = 0.0002) -> dict:
    """Submit `requests` ({"rid", "at", "ids", "n_out"}) at their times (or one after another when not open loop)
    and collect tokens with the time poll() first returned them."""
    pending = sorted(requests, key=lambda r: r["at"])
    records = {r["rid"]: {"rid": r["rid"], "n_in": len(r["ids"]), "n_out": r["n_out"], "planned_at": r["at"],
                          "t_submit": None, "token_times": [], "tokens": [], "finished": False, "submit_seconds": None}
               for r in requests}
    t0 = time.perf_counter()
    last_arrival = pending[-1]["at"] if (pending and open_loop) else 0.0
    index, running, poll_calls, poll_max = 0, 0, 0, 0.0
    deadline = None
    while True:
        now = time.perf_counter() - t0
        while index < len(pending) and (pending[index]["at"] <= now if open_loop else running == 0):
            req = pending[index]
            a = time.perf_counter()
            engine.submit(req["rid"], req["ids"], req["n_out"])
            b = time.perf_counter()
            records[req["rid"]].update(t_submit=a - t0, submit_seconds=b - a)
            index += 1
            running += 1
            if not open_loop:
                break
        a = time.perf_counter()
        events = engine.poll()
        b = time.perf_counter()
        poll_calls += 1
        poll_max = max(poll_max, b - a)
        for rid, tokens, finished in events:
            rec = records[rid]
            rec["tokens"].extend(int(t) for t in tokens)
            rec["token_times"].extend([b - t0] * len(tokens))
            if finished and not rec["finished"]:
                rec["finished"] = True
                running -= 1
        if index == len(pending) and running == 0:
            break
        if open_loop and index == len(pending):
            deadline = deadline or (max(last_arrival, time.perf_counter() - t0) + drain_s)
            if time.perf_counter() - t0 > deadline:
                break
        if not events:
            time.sleep(poll_sleep)
    return {"records": list(records.values()), "wall_s": time.perf_counter() - t0, "poll_calls": poll_calls,
            "poll_max_ms": poll_max * 1e3, "last_arrival_s": last_arrival}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--plan", required=True, help="JSON: {'requests': [{rid, at, ids, n_out}], 'probe': [rid, ...]}")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=34816)
    parser.add_argument("--drain-s", type=float, default=90.0)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    t0 = time.time()
    engine = load_engine(args.workspace, args.model_dir, args.max_batch, args.max_context)
    report: dict = {"load_seconds": time.time() - t0}
    missing = [m for m in ("submit", "poll", "idle", "kv_pages", "drop_cache") if not hasattr(engine, m)]
    if missing:
        report["missing_methods"] = missing
        Path(args.out).write_text(json.dumps(report) + "\n")
        return 0

    def pages() -> list[int]:
        free, total = engine.kv_pages()
        return [int(free), int(total)]

    report["pages_at_start"] = pages()
    for name, requests in plan["replays"].items():
        engine.drop_cache()
        result = drive(engine, requests, open_loop=True, drain_s=args.drain_s)
        result["idle_after"] = bool(engine.idle())
        result["pages_after"] = pages()
        if hasattr(engine, "stats"):
            try:
                result["engine_stats"] = engine.stats()
            except Exception as error:
                result["engine_stats_error"] = f"{type(error).__name__}: {error}"
        # Let whatever is still running finish before the next phase, off the record (bounded).
        stop = time.time() + 300
        while not engine.idle() and time.time() < stop:
            engine.poll()
        engine.drop_cache()
        result["pages_after_drop"] = pages()
        report.setdefault("replays", {})[name] = result
    # The probes alone, one at a time, with an empty cache: the reference for "the same tokens under load".
    by_rid = {r["rid"]: r for requests in plan["replays"].values() for r in requests}
    alone = {}
    for rid in plan.get("probe", []):
        engine.drop_cache()
        req = dict(by_rid[rid], at=0.0)
        alone[str(rid)] = drive(engine, [req], open_loop=False, drain_s=args.drain_s)["records"][0]
    report["alone"] = alone
    # The offline reference prompts through the online path (compared by the parent with the step-by-step tokens).
    if plan.get("reference_prompts"):
        refs = [{"rid": 10_000_000 + i, "at": 0.0, "ids": ids, "n_out": plan["reference_new_tokens"]}
                for i, ids in enumerate(plan["reference_prompts"])]
        engine.drop_cache()
        report["reference_alone"] = [drive(engine, [r], open_loop=False, drain_s=args.drain_s)["records"][0]["tokens"] for r in refs]
        engine.drop_cache()
        report["reference_together"] = [rec["tokens"] for rec in sorted(
            drive(engine, refs, open_loop=True, drain_s=args.drain_s)["records"], key=lambda r: r["rid"])]
    engine.drop_cache()
    report["pages_at_end"] = pages()
    if hasattr(engine, "close"):
        engine.close()
    Path(args.out).write_text(json.dumps(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
