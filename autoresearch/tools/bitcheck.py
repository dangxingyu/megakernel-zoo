#!/usr/bin/env python3
"""Are two builds of an engine the same function? Logits of build A and build B on the same teacher-forced tokens,
compared bit for bit and numerically. Use it before calling a track-B change "bit-identical to track A", and to
check that a refactor changed nothing.

    python tools/bitcheck.py --a WS_A --b WS_B [--positions 1152] [--batch 1 8] [--every 4] [--seed 0]
    python tools/bitcheck.py --a . --a-track A --b . --b-track "A W12"     # same source, two TRACK values

WS_A / WS_B are workspace directories with engine/ (build.sh is run in a private copy of each). --a-track/--b-track
overwrite engine/TRACK in that copy. Each build runs in its own process: reset(), then step() on random token ids
(seeded; the same for A and B) for --positions positions, batch sizes from --batch; the logits of every --every-th
position are kept (float32, on the host) and compared:
  bit_identical   share of kept (position, row) logit vectors that are equal bit for bit
  first_diff      first (batch, position) whose logits differ
  max_abs         largest |A - B| over all kept logits; rel_l2 median / max over positions (centred logits)
  top1            share of positions where A and B choose the same token
Also within each build: batch invariance (row 0 of the largest batch vs batch 1 with the same tokens).
Prints one BITCHECK json line; exit code 0 when every comparison is bit-identical, 1 otherwise.
Needs: $MK_MODEL_DIR (the model), a GPU (run it through `mk exec`).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def dump(ws: str, out: str, positions: int, batches: list[int], every: int, seed: int) -> None:
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(ws) / "engine"))
    spec = importlib.util.spec_from_file_location("engine_api_bitcheck", Path(ws) / "engine" / "api.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = json.load(open(Path(os.environ["MK_MODEL_DIR"]) / "config.json"))
    vocab = cfg.get("vocab_size") or cfg.get("text_config", {}).get("vocab_size")
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, min(vocab, 100000), (max(batches), positions), generator=g)
    eng = mod.Engine(os.environ["MK_MODEL_DIR"], max(batches), positions + 8)
    res = {}
    with torch.no_grad():
        for b in batches:
            eng.reset()
            ids = tokens[:b].cuda()
            kept = []
            for p in range(positions):
                logits = eng.step(ids[:, p].contiguous())
                if p % every == 0 or p == positions - 1:
                    kept.append(logits.float().cpu().numpy())
            res[f"b{b}"] = np.stack(kept, 1)  # [b, kept positions, vocab]
    np.savez(out, **res)
    if hasattr(eng, "close"):
        eng.close()


def compare(a, b):
    import numpy as np
    same = np.all(a.view(np.uint32) == b.view(np.uint32), axis=-1)  # [rows, positions]
    first = None
    if not same.all():
        r, p = np.argwhere(~same)[0]
        first = [int(r), int(p)]
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    ca = a - a.mean(-1, keepdims=True); cb = b - b.mean(-1, keepdims=True)
    rel = np.linalg.norm(ca - cb, axis=-1) / np.maximum(np.linalg.norm(cb, axis=-1), 1e-30)
    return {"bit_identical": float(same.mean()), "first_diff": first, "max_abs": float(np.nanmax(d)),
            "rel_l2_median": float(np.median(rel)), "rel_l2_max": float(np.max(rel)),
            "top1": float((a.argmax(-1) == b.argmax(-1)).mean()), "finite": bool(np.isfinite(a).all() and np.isfinite(b).all())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--a-track"); ap.add_argument("--b-track")
    ap.add_argument("--positions", type=int, default=1152)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--_dump", nargs=2, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._dump:
        dump(args._dump[0], args._dump[1], args.positions, args.batch, args.every, args.seed)
        return 0
    import numpy as np
    tmp = Path(tempfile.mkdtemp(prefix="bitcheck-"))
    outs = {}
    for side, ws, track in (("a", args.a, args.a_track), ("b", args.b, args.b_track)):
        copy = tmp / side
        shutil.copytree(Path(ws) / "engine", copy / "engine", symlinks=True)
        if track is not None:
            (copy / "engine" / "TRACK").write_text(track + "\n")
        if (copy / "engine" / "build.sh").exists():
            r = subprocess.run(["bash", "engine/build.sh"], cwd=copy, capture_output=True, text=True)
            if r.returncode:
                print(f"build {side} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
                return 2
        out = tmp / f"{side}.npz"
        r = subprocess.run([sys.executable, __file__, "--a", "-", "--b", "-", "--positions", str(args.positions), "--batch",
                            *map(str, args.batch), "--every", str(args.every), "--seed", str(args.seed),
                            "--_dump", str(copy), str(out)], cwd=copy)
        if r.returncode:
            return 2
        outs[side] = np.load(out)
    report = {"positions": args.positions, "every": args.every, "batches": args.batch, "a": args.a, "b": args.b,
              "a_track": args.a_track, "b_track": args.b_track}
    ok = True
    for key in outs["a"].files:
        c = compare(outs["a"][key], outs["b"][key])
        report[f"a_vs_b_{key}"] = c
        ok &= c["bit_identical"] == 1.0
    big = f"b{max(args.batch)}"
    if "b1" in outs["a"].files and big != "b1":
        for side in ("a", "b"):
            report[f"batch_invariance_{side}"] = compare(outs[side][big][:1], outs[side]["b1"])
    print("BITCHECK " + json.dumps(report))
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
