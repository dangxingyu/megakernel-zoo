#!/usr/bin/env python
"""Does an engine's multi-token (draft-and-verify) path produce exactly its own token-by-token greedy tokens, and how
repetitive is the text it is measured on?

    python eval/spec_check.py gen --path step|online --workspace WS --model-dir D --out F.json [--n 1024] [--prompts 4]
                                  [--prompts-file natural.json] [--batch B] [--stagger S] [--max-batch M]
    python eval/spec_check.py compare STEP.json ONLINE.json

`gen` decodes the evaluation's own kind of prompt (eval/prompts.make, 64 tokens, seeds 0..) for n new tokens, one
sequence at a time: path `step` through reset()/step()/argmax (one token per model pass: the exact reference),
path `online` through submit()/poll() (whatever the engine does there, e.g. drafting and verifying several tokens).
Each path runs in its own process. `compare` reports per prompt whether the two token lists are identical, the
first position where they differ, and how copyable the reference text is: the share of positions whose token is
the one that followed the previous occurrence of the same 4 preceding tokens (what a context-lookup drafter gets
right), and the length of the final repeating cycle if the text has fallen into a loop.
"""
import json
import sys
import time
from pathlib import Path


def gen(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _device
    import prompts
    from _engine_runner import load_engine

    D = _device.get()  # cuda: torch tensors; metal (macOS): NumPy tokens, MLX logits (eval/_device.py)

    tok = prompts.load(args.model_dir)
    engine = load_engine(args.workspace, args.model_dir, args.max_batch or max(8, args.batch), 2048)
    if args.prompts_file:  # e.g. natural text (eval/natural_prompts.py): a JSON list of token-id lists
        plist = [list(map(int, p)) for p in json.loads(Path(args.prompts_file).read_text())][:args.prompts]
    else:
        plist = [prompts.make(tok, 64, seed)["ids"] for seed in range(args.prompts)]
    out = {"path": args.path, "n": args.n, "batch": args.batch, "stagger": args.stagger,
           "prompts_file": args.prompts_file, "prompts": []}
    B = args.batch
    with D.no_grad():
        for g0 in range(0, len(plist), B):
            group = plist[g0:g0 + B]
            t0 = time.perf_counter()
            if args.path == "step":
                # step()/argmax, the group decoded together in lockstep (one token per model pass: the exact reference
                # for the engine at this batch composition); prompts of one group must have the same length
                assert len({len(p) for p in group}) == 1, "a step group needs prompts of equal length"
                engine.reset()
                for i in range(len(group[0]) - 1):
                    D.launch(engine.step(D.tensor([p[i] for p in group])))
                last, produced = D.tensor([p[-1] for p in group]), [[] for _ in group]
                for _ in range(args.n):
                    last = D.argmax(engine.step(last))
                    for j, t in enumerate(D.tolist(last)):
                        produced[j].append(int(t))
                bursts_mean = [None] * len(group)
            else:
                # submit()/poll(): the whole group at once (stagger 0), or the next request only after the previous one
                # has received `stagger` tokens (requests join a running cohort)
                produced, bursts, done = [[] for _ in group], [[] for _ in group], [False] * len(group)
                rid0 = 1000 * (g0 + 1)
                submitted = 0
                if args.stagger <= 0:
                    for j, p in enumerate(group):
                        engine.submit(rid0 + j, p, args.n)
                    submitted = len(group)
                else:
                    engine.submit(rid0, group[0], args.n); submitted = 1
                polls = 0
                while not all(done[:submitted]) or submitted < len(group):
                    for rid, tokens, finished in engine.poll():
                        j = rid - rid0
                        produced[j].extend(int(t) for t in tokens)
                        if len(tokens):
                            bursts[j].append(len(tokens))
                        done[j] = done[j] or bool(finished)
                    if submitted < len(group) and (len(produced[submitted - 1]) >= args.stagger or done[submitted - 1]):
                        engine.submit(rid0 + submitted, group[submitted], args.n); submitted += 1
                    polls += 1
                    if polls > 50_000_000:
                        break
                bursts_mean = [sum(b) / max(1, len(b)) for b in bursts]
            dt = time.perf_counter() - t0
            for j, p in enumerate(group):
                out["prompts"].append({"seed": g0 + j, "prompt": p, "tokens": produced[j], "seconds": dt, "bursts_mean": bursts_mean[j]})
            print(f"MK_SPECCHECK_PROGRESS {args.path} b{B} group {g0 // B}: {sum(map(len, produced))} tokens in {dt:.1f} s", flush=True)
    bm = [r["bursts_mean"] for r in out["prompts"] if r["bursts_mean"] is not None]
    if bm:
        out["bursts_mean"] = bm
    Path(args.out).write_text(json.dumps(out))
    return 0


def copyable(tokens: list[int], k: int = 4) -> float:
    seen, hits = {}, 0
    for i in range(k, len(tokens)):
        key = tuple(tokens[i - k:i])
        if key in seen and seen[key] == tokens[i]:
            hits += 1
        seen[key] = tokens[i]
    return hits / max(1, len(tokens) - k)


def cycle(tokens: list[int]) -> int:
    """Shortest period p (<= 256) such that the last 4p tokens repeat with period p; 0 if none."""
    for p in range(1, 257):
        tail = tokens[-4 * p:]
        if len(tail) == 4 * p and all(tail[i] == tail[i - p] for i in range(p, len(tail))):
            return p
    return 0


def compare(a_path: str, b_path: str) -> int:
    a, b = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    rows = []
    for x, y in zip(a["prompts"], b["prompts"]):
        m = min(len(x["tokens"]), len(y["tokens"]))
        first = next((i for i in range(m) if x["tokens"][i] != y["tokens"][i]), None)
        rows.append({"seed": x["seed"], "identical": first is None and len(x["tokens"]) == len(y["tokens"]), "first_difference": first,
                     "tokens": [len(x["tokens"]), len(y["tokens"])], "copyable_share_reference": round(copyable(x["tokens"]), 3),
                     "final_cycle_length": cycle(x["tokens"]), "distinct_tokens_reference": len(set(x["tokens"])),
                     "seconds": [round(x["seconds"], 1), round(y["seconds"], 1)]})
    print("MK_SPECCHECK " + json.dumps({"reference": a["path"], "other": b["path"], "n": a["n"], "identical_prompts": sum(r["identical"] for r in rows),
                                        "prompts": len(rows), "mean_tokens_per_poll_burst": b.get("bursts_mean") or a.get("bursts_mean"), "rows": rows}), flush=True)
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["compare"]:
        raise SystemExit(compare(sys.argv[2], sys.argv[3]))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["gen"])
    parser.add_argument("--path", choices=["step", "online"], required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--prompts-file", default=None, help="JSON list of token-id lists (default: eval/prompts.make, seeds 0..)")
    parser.add_argument("--batch", type=int, default=1, help="sequences decoded together (step: lockstep; online: submitted together)")
    parser.add_argument("--stagger", type=int, default=0, help="online: submit the next request after the previous one has this many tokens")
    parser.add_argument("--max-batch", type=int, default=0, help="max_batch the engine is built with (default max(8, --batch); "
                                                                   "1 for an engine that serves one sequence, e.g. on a Mac)")
    raise SystemExit(gen(parser.parse_args()))
