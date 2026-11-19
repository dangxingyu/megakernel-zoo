#!/usr/bin/env python
"""Child process of kernel_eval.py: loads the engine under test and drives it. Not called by hand.

    python eval/_engine_runner.py --workspace WS --model-dir DIR --ref REF --mode check|time|trace --out FILE ...

It only produces raw material (logits, wall times); every comparison with the reference and every verdict is
made by the parent, so that the code under test never grades itself.

The engine contract (WS/engine/api.py):

    class Engine:
        def __init__(self, model_dir: str, max_batch: int, max_context: int): ...
        def reset(self) -> None:            # forget every sequence
        def step(self, tokens):             # tokens: int64 CUDA tensor [B], the next input token of each sequence
            ...                             # (B is fixed between resets); appends them and returns the logits
            return logits                   # of the following token: CUDA tensor [B, vocab], bf16 or float32
        def close(self) -> None: ...        # optional
        def stats(self) -> dict: ...        # optional: counters the engine wants on the record
        def generate(self, tokens, n_steps): ...   # optional: greedy decode inside the engine. tokens int64 [B] is the
            return new_tokens                      # next input token; returns int64 cuda [B, n_steps] (the n_steps tokens
                                                   # chosen), sampling, feedback and the loop all on the device
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def load_engine(workspace: str, model_dir: str, max_batch: int, max_context: int):
    api = Path(workspace) / "engine" / "api.py"
    sys.path.insert(0, str(api.parent))
    spec = importlib.util.spec_from_file_location("engine_api", api)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Engine(model_dir, max_batch, max_context)


def watch_for_stall(engine, stall_s: float) -> None:
    """End this process when no call into the engine has returned for stall_s seconds.

    A deadlocked engine (cards waiting on each other, a host thread blocked in a synchronize) otherwise holds the
    machine until the stage's timeout, hours on a large model. Started after the engine is loaded: its methods are
    wrapped to record the time of each return (one extra Python call per engine call), and a thread checks it."""
    import functools
    import threading
    import traceback

    last = [time.time()]
    for name in ("step", "generate", "reset"):
        method = getattr(engine, name, None)
        if callable(method):
            def counted(*a, _m=method, **k):
                out = _m(*a, **k)
                last[0] = time.time()
                return out
            setattr(engine, name, functools.wraps(method)(counted))
    main_id = threading.main_thread().ident

    def loop():
        while True:
            time.sleep(min(10.0, stall_s / 4))
            if time.time() - last[0] > stall_s:
                frame = sys._current_frames().get(main_id)
                print(f"[runner] no engine call returned for {stall_s:.0f} s; the engine is probably deadlocked. "
                      "The main thread is at:\n" + "".join(traceback.format_stack(frame)[-6:]), file=sys.stderr, flush=True)
                os._exit(3)

    threading.Thread(target=loop, daemon=True, name="stall-watch").start()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--mode", choices=["check", "time", "trace"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-sequences", type=int, default=4)
    parser.add_argument("--warm-steps", type=int, default=64)
    parser.add_argument("--timed-steps", type=int, default=256)
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--w1-steps", type=int, default=1024, help="time mode: length of the W1-like greedy decode (0 = skip)")
    args = parser.parse_args()
    ref = Path(args.ref)
    ids = torch.from_numpy(np.load(ref / "ids.npy")).cuda()  # [R, T_long]
    meta = json.loads((ref / "meta.json").read_text())
    R, T = ids.shape[0], meta["check_tokens"]
    out = Path(args.out)
    t0 = time.time()
    engine = load_engine(args.workspace, args.model_dir, max_batch=R, max_context=args.max_context)
    if float(os.environ.get("MK_EVAL_STALL_S") or 1200) > 0:
        watch_for_stall(engine, float(os.environ.get("MK_EVAL_STALL_S") or 1200))
    report: dict = {"mode": args.mode, "load_seconds": time.time() - t0}

    with torch.no_grad():
        if args.mode == "check":
            # (a) each of the first sequences alone, every position; (b) all R together, every position.
            single = torch.empty(args.check_sequences, T, meta["vocab"], dtype=torch.float32)
            for r in range(args.check_sequences):
                engine.reset()
                for t in range(T):
                    single[r, t] = engine.step(ids[r:r + 1, t])[0].float().cpu()
            engine.reset()
            batched = torch.empty(R, T, meta["vocab"], dtype=torch.float32)
            for t in range(T):
                batched[:, t] = engine.step(ids[:, t]).float().cpu()
            # (c) free-running greedy from the prompt, all R together: what a user would get.
            engine.reset()
            P = meta["prompt_tokens"]
            for t in range(P - 1):
                engine.step(ids[:, t])
            token = ids[:, P - 1]
            free = []
            for _ in range(T - P):
                token = engine.step(token).float().argmax(-1)
                free.append(token.cpu())
            if (ref / "long_positions.npy").exists():
                # (b') all R together along the whole long sequence, logits kept at the reference's long positions.
                wanted = {int(t): i for i, t in enumerate(np.load(ref / "long_positions.npy"))}
                long = torch.empty(R, len(wanted), meta["vocab"], dtype=torch.float32)
                engine.reset()
                for t in range(ids.shape[1]):
                    logits = engine.step(ids[:, t])
                    if t in wanted:
                        long[:, wanted[t]] = logits.float().cpu()
                np.save(out.with_suffix(".long.npy"), long.numpy())
            np.save(out.with_suffix(".single.npy"), single.numpy())
            np.save(out.with_suffix(".batched.npy"), batched.numpy())
            np.save(out.with_suffix(".free.npy"), torch.stack(free, dim=1).numpy())
            if hasattr(engine, "generate"):
                # (d) the same continuation produced inside the engine: must be what (c) produced step by step.
                engine.reset()
                for t in range(P - 1):
                    engine.step(ids[:, t])
                np.save(out.with_suffix(".gen.npy"), engine.generate(ids[:, P - 1], T - P).cpu().numpy())
                report["has_generate"] = True
        else:
            batches = [1] if args.mode == "trace" else [int(b) for b in args.batches.split(",")]
            segments = 1 if args.mode == "trace" else args.segments
            warm, timed = args.warm_steps, args.timed_steps
            if warm + timed > ids.shape[1]:
                raise SystemExit("reference sequences are too short for warm + timed steps")
            known = set(range(T)) | ({int(t) for t in np.load(ref / "long_positions.npy")} if (ref / "long_positions.npy").exists() else set())
            inside = sorted(t for t in known if warm <= t < warm + timed)
            probe_steps = sorted({inside[i * (len(inside) - 1) // 5] for i in range(6)}) if inside else []
            report["timing"] = {}
            for B in batches:
                walls, probes, finite = [], {}, False
                # One extra segment first, not recorded: the first pass after a load pays one-time costs
                # (the self-test's first segment took 2.4x the others).
                for segment in range(segments + (0 if args.mode == "trace" else 1)):
                    engine.reset()
                    for t in range(warm):
                        engine.step(ids[:B, t])
                    torch.cuda.synchronize()
                    kept = {}
                    if args.mode == "trace":
                        torch.cuda.nvtx.range_push("mk_timed")
                    start = time.perf_counter()
                    for t in range(warm, warm + timed):
                        logits = engine.step(ids[:B, t])
                        if t in probe_steps:
                            kept[t] = logits  # compared by the parent after the clock stops: timed steps are real steps
                    torch.cuda.synchronize()
                    wall = time.perf_counter() - start
                    if args.mode == "trace":
                        torch.cuda.nvtx.range_pop()
                    if args.mode == "trace" or segment > 0:
                        walls.append(wall)
                    if args.mode == "trace" or segment > 0:
                        for t, value in kept.items():  # every recorded segment is checked, not only the last
                            np.save(out.with_suffix(f".probe.b{B}.s{segment}.t{t}.npy"), value.float().cpu().numpy())
                    probes = {t: None for t in kept}
                    finite = bool(torch.isfinite(logits.float()).all())
                report["timing"][str(B)] = {"walls_s": walls, "steps": timed, "warm_steps": warm, "last_logits_finite": finite,
                                            "context_start": warm, "context_end": warm + timed, "probe_steps": sorted(probes)}
                report["timing"][str(B)]["probe_segments"] = list(range(1, segments + 1)) if args.mode != "trace" else [0]
                if hasattr(engine, "generate"):
                    walls = []
                    for segment in range(segments + (0 if args.mode == "trace" else 1)):
                        engine.reset()
                        for t in range(warm):
                            engine.step(ids[:B, t])
                        torch.cuda.synchronize()
                        if args.mode == "trace":
                            torch.cuda.nvtx.range_push("mk_generate")
                        start = time.perf_counter()
                        produced = engine.generate(ids[:B, warm], timed)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        if args.mode == "trace":
                            torch.cuda.nvtx.range_pop()
                        if args.mode == "trace" or segment > 0:
                            walls.append(elapsed)
                    report["timing"][str(B)]["generate_walls_s"] = walls
                    report["timing"][str(B)]["generate_shape_ok"] = tuple(produced.shape) == (B, timed)
                if args.mode == "time" and args.w1_steps:
                    # W1-like: what the baselines' workload does, without HTTP: a 64-token prompt, then w1_steps greedy
                    # tokens, the chosen token copied to the host at every step (a host-driven server needs it there
                    # to stream it and to test for the end of the sequence). Context runs from 64 to 64 + w1_steps.
                    walls = []
                    for segment in range(2):
                        engine.reset()
                        for t in range(63):
                            engine.step(ids[:B, t])
                        token = ids[:B, 63]
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        for _ in range(args.w1_steps):
                            token = engine.step(token).argmax(-1)
                            token.cpu()
                        elapsed = time.perf_counter() - start
                        if segment > 0:
                            walls.append(elapsed)
                    report["timing"][str(B)].update(w1_walls_s=walls, w1_steps=args.w1_steps)
                    if hasattr(engine, "generate"):
                        # The same workload through the engine's own loop: tokens reach the host when it returns.
                        walls = []
                        for segment in range(2):
                            engine.reset()
                            for t in range(63):
                                engine.step(ids[:B, t])
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            produced = engine.generate(ids[:B, 63], args.w1_steps).cpu()
                            elapsed = time.perf_counter() - start
                            if segment > 0:
                                walls.append(elapsed)
                        report["timing"][str(B)].update(w1_generate_walls_s=walls)
    if hasattr(engine, "stats"):
        try:
            report["engine_stats"] = engine.stats()
        except Exception as error:
            report["engine_stats_error"] = f"{type(error).__name__}: {error}"
    try:
        import pynvml
        pynvml.nvmlInit()
        card = int(os.environ.get("MK_CARD", "0") or 0)  # NVML counts all cards of the pod, whatever CUDA may see
        card = card if card < pynvml.nvmlDeviceGetCount() else 0
        report["gpu_memory_used_gb"] = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(card)).used / 1e9
        report["card"] = card
        # every card of the pod (a whole-node engine spreads over them; a one-card engine leaves the others empty)
        report["gpu_memory_used_gb_per_card"] = [round(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).used / 1e9, 2)
                                                 for i in range(pynvml.nvmlDeviceGetCount())]
    except Exception:
        pass
    if hasattr(engine, "close"):
        engine.close()
    out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
