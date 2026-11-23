#!/usr/bin/env python
"""Checks of a finished engine beyond the run's evaluator, against the same float64 reference. One process per call.

    python eval/release_check.py --workspace WS --model-dir DIR --ref REF --out OUT.json [--max-context 2048]
    python eval/release_check.py --compare A.json B.json     # two processes of the same engine: same outputs?

What the run's evaluator already checks: batch 1 and batch R (= 8) against float64 at 128 positions, R together along
1152 tokens, generate() against step(). What this adds, teacher-forced on the reference's sequences:

  every_batch   each batch size b = 1..R (the sequences b-1 .. 2b-2 of the reference, so that every size sees other
                sequences): error against float64 (the evaluator's limit: TOLERANCE x the BF16 path's own error) and
                top-1 agreement with float64
  slots         the R sequences in reversed slot order: the logits of a sequence must not depend on its slot
                (difference to the forward order, as relative L2 and as whether it is bit-identical)
  neighbours    sequence 0 at batch R with different neighbours (the other slots hold other sequences): how much its
                logits move with what else is in the batch, compared with alone
  repeat        batch 1 and batch R a second time after reset(): bit-identical to the first time?
  after_long    the engine is driven to its context limit (max_context steps at batch R), then reset() and batch 1
                again: bit-identical to before? (state left behind by a long request)
  digest        sha256 of the greedy tokens (128 steps, batch 1 and R) and of the batch-1 logits, for --compare

Raw numbers and verdicts are computed here, not by the engine.

--max-batch M (default R; 1 on Apple silicon, MK_DEVICE=metal, where an engine may serve one sequence only): batch
sizes 1..M are checked, `slots` and `neighbours` need M > 1 and are skipped otherwise, and `repeat`, `after_long` and
`digest` run at batch M instead of R.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _device  # noqa: E402
from _engine_runner import load_engine, watch_for_stall  # noqa: E402

TOLERANCE = 2.0  # as eval/kernel_eval.py


def rel_l2(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    a, ref = a.astype(np.float64), ref.astype(np.float64)
    return np.linalg.norm(a - ref, axis=-1) / np.linalg.norm(ref, axis=-1)


def stats(x: np.ndarray) -> dict:
    return {"median": float(np.median(x)), "max": float(np.max(x))}


def compare(a_path: str, b_path: str) -> int:
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    same = {k: a["digest"][k] == b["digest"][k] for k in a["digest"]}
    print("MK_RELCHECK " + json.dumps({"check": "restart", "same": same, "pass": all(same.values())}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--workspace")
    ap.add_argument("--model-dir")
    ap.add_argument("--ref")
    ap.add_argument("--out")
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--positions", type=int, default=64, help="teacher-forced positions per batch size")
    ap.add_argument("--max-batch", type=int, default=0, help="largest batch checked (default R; 1 on metal)")
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    D = _device.get()
    ref = Path(args.ref)
    if not (ref / "meta.json").exists() or not (ref / "logits64.npy").exists():
        raise SystemExit(f"the reference in {ref} is incomplete (meta.json, logits64.npy): point --ref at the float64 reference")
    meta = json.loads((ref / "meta.json").read_text())
    ids = D.ids(np.load(ref / "ids.npy"))                               # [R, T_long]
    ref_logits = np.load(ref / "logits64.npy", mmap_mode="r")           # [R, T, V]
    R, T = ids.shape[0], min(args.positions, meta["check_tokens"])
    RB = min(args.max_batch or (1 if D.name == "metal" else R), R)      # the largest batch checked (R on cuda)
    limit = {k: TOLERANCE * meta["hf_bf16_rel_l2"][k] for k in ("median", "max")}
    t0 = time.time()
    engine = load_engine(args.workspace, args.model_dir, max_batch=RB, max_context=args.max_context)
    watch_for_stall(engine, 900.0 if D.name == "cuda" else 3600.0)
    report: dict = {"load_seconds": time.time() - t0, "R": R, "max_batch": RB, "device": D.name, "positions": T, "limit": limit, "checks": {}}

    def forced(rows: list[int], steps: int = T) -> np.ndarray:
        engine.reset()
        out = np.empty((len(rows), steps, meta["vocab"]), dtype=np.float32)
        sel = ids[rows]
        for t in range(steps):
            out[:, t] = D.host(engine.step(sel[:, t]))
        return out

    def greedy(rows: list[int], steps: int = 128) -> np.ndarray:
        engine.reset()
        P, sel = meta["prompt_tokens"], ids[rows]
        for t in range(P - 1):
            engine.step(sel[:, t])
        token, out = sel[:, P - 1], []
        for _ in range(steps):
            token = D.argmax(engine.step(token))
            out.append(D.tokens_host(token))
        return np.stack(out, axis=1)

    def emit(name: str, body: dict) -> None:
        report["checks"][name] = body
        print("MK_RELCHECK " + json.dumps({"check": name, **body}), flush=True)

    with D.no_grad():
        runs: dict[int, tuple[list[int], np.ndarray]] = {}
        per_batch = {}
        for b in range(1, RB + 1):
            rows = [(b - 1 + k) % R for k in range(b)]
            try:
                out = forced(rows)
            except Exception as error:  # a batch size the engine refuses
                per_batch[b] = {"pass": False, "error": f"{type(error).__name__}: {str(error)[:200]}"}
                continue
            runs[b] = (rows, out)
            target = np.asarray(ref_logits[rows, :T])
            err = stats(rel_l2(out, target)) if np.isfinite(out).all() else {"median": float("inf"), "max": float("inf")}
            agree = float((out.argmax(-1) == target.argmax(-1)).mean())
            per_batch[b] = {"pass": bool(err["median"] <= limit["median"] and err["max"] <= limit["max"]), "rel_l2": err,
                            "top1_agreement_with_float64": agree, "rows": rows}
        emit("every_batch", {"pass": all(v["pass"] for v in per_batch.values()), "by_batch": per_batch})

        if RB in runs:
            rows, forward = runs[RB]
            if RB == 1:
                alone = forced([0])
            else:
                back = forced(rows[::-1])[::-1]
                d = rel_l2(back, forward)
                emit("slots", {"pass": bool(d.max() <= limit["max"]), "bit_identical": bool(np.array_equal(back, forward)),
                               "rel_l2": stats(d)})
                # sequence 0: alone, with the reference's neighbours, with every neighbour replaced by sequence 0 itself
                alone = forced([0])
                pos0 = rows.index(0)
                with_others = forward[pos0:pos0 + 1]
                with_self = forced([0] * RB)[:1]
                emit("neighbours", {"pass": bool(rel_l2(with_others, alone).max() <= limit["max"]),
                                    "alone_vs_in_batch": stats(rel_l2(with_others, alone)),
                                    "alone_vs_batch_of_copies": stats(rel_l2(with_self, alone)),
                                    "copies_agree_with_each_other": bool(np.array_equal(forced([0] * RB)[0], forced([0] * RB)[RB - 1])),
                                    "bit_identical_alone_vs_in_batch": bool(np.array_equal(with_others, alone))})
            again1, againR = forced([0]), forced(rows)
            emit("repeat", {"pass": bool(np.array_equal(again1, alone) and np.array_equal(againR, forward)),
                            "batch1_bit_identical": bool(np.array_equal(again1, alone)),
                            "batchR_bit_identical": bool(np.array_equal(againR, forward))})
            # drive the engine to its context limit, then check that a fresh request is unaffected
            engine.reset()
            steps_done, finite, error = 0, True, None
            try:
                for t in range(args.max_context):
                    logits = engine.step(ids[:RB, t % ids.shape[1]])
                    D.launch(logits)
                    steps_done += 1
                    if t % 256 == 255 or t == args.max_context - 1:
                        finite = finite and D.finite(logits)
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)[:200]}"
            after = forced([0])
            emit("after_long", {"pass": bool(error is None and finite and np.array_equal(after, alone)),
                                "steps_at_batch_R": steps_done, "batch": RB, "max_context": args.max_context, "finite": finite, "error": error,
                                "batch1_after_bit_identical": bool(np.array_equal(after, alone))})
            g1, gR = greedy([0]), greedy(rows)
            report["digest"] = {"greedy_b1": hashlib.sha256(g1.tobytes()).hexdigest()[:16],
                                "greedy_bR": hashlib.sha256(gR.tobytes()).hexdigest()[:16],
                                "logits_b1": hashlib.sha256(alone.tobytes()).hexdigest()[:16]}
            print("MK_RELCHECK " + json.dumps({"check": "digest", **report["digest"]}), flush=True)
    report["seconds"] = time.time() - t0
    Path(args.out).write_text(json.dumps(report, indent=1))
    close = getattr(engine, "close", None)
    if callable(close):
        close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
