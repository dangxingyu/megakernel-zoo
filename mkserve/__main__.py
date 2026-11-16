#!/usr/bin/env python
"""mkserve: serve a model through its megakernel engine behind the OpenAI API, the way `vllm serve` is used.

    mkserve list                      (or: python -m mkserve list)
    mkserve serve Qwen/Qwen3-8B [--engine NAME | --track A|B] [--port 8000] [--host 127.0.0.1]
                                          [--model-dir DIR] [--mode auto|native|step] [--workdir DIR]

`serve` finds the engine written for this model on this card (mkserve/registry.json), unpacks and builds it (nvcc
must be on PATH), takes the weights from --model-dir or downloads the Hugging Face snapshot, starts the engine worker
and the HTTP front end, and stays in the foreground until interrupted. Then:

    curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \\
         -d '{"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'

Needs: one supported NVIDIA card, the CUDA toolkit, Python with torch, numpy, aiohttp, tokenizers, jinja2 (and
huggingface_hub when the weights are to be downloaded). An engine exists only for the (model, card) pairs listed by
`list`: there is no generic fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CARDS = (("H100", "h100"), ("GB200", "b200"), ("B200", "b200"))  # substring of nvidia-smi's name -> registry key


def detect_card() -> tuple[str | None, str]:
    try:
        name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.splitlines()[0].strip()
    except Exception:
        return None, "no GPU found (nvidia-smi failed)"
    return next((key for sub, key in CARDS if sub in name), None), name


def cmd_list(_args) -> int:
    reg = json.loads((HERE / "registry.json").read_text())
    for model, cards in reg["models"].items():
        for card, entry in cards.items():
            for name, e in entry["engines"].items():
                mark = "*" if name == entry["default"] else " "
                print(f"{mark} {model:28s} {card:6s} {name:10s} track {e['track']}  {e['kind']:8s} {e['what']}")
    print("* = the default engine of that model and card")
    return 0


def is_product(rel: Path) -> bool:
    """Files a build or a run leaves in an engine's folder: not part of its source."""
    return rel.suffix in (".so", ".o", ".pyc") or any(part in ("__pycache__", "build") for part in rel.parts)


def source_digest(source: Path) -> str:
    """sha256 of an engine source: of the archive, or for a directory of every file's path and bytes in sorted order."""
    if not source.is_dir():
        return hashlib.sha256(source.read_bytes()).hexdigest()
    h = hashlib.sha256()
    for f in sorted(p for p in source.rglob("*") if p.is_file() and not is_product(p.relative_to(source))):
        h.update(f.relative_to(source).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def cmd_serve(args) -> int:
    reg = json.loads((HERE / "registry.json").read_text())
    cards = reg["models"].get(args.model)
    if cards is None:
        print(f"mkserve: no megakernel engine is registered for {args.model}. Registered: {', '.join(reg['models'])}", file=sys.stderr)
        return 2
    card, card_name = (args.card, args.card) if args.card else detect_card()
    if card not in cards:
        print(f"mkserve: {args.model} has engines for {', '.join(cards)}; this machine's card is '{card_name}'. An engine is written for one card type.", file=sys.stderr)
        return 2
    entry = cards[card]
    name = args.engine or next((n for n, e in entry["engines"].items() if args.track and e["track"] == args.track.upper()
                                and e["kind"] == entry["engines"][entry["default"]]["kind"]), None) or entry["default"]
    if name not in entry["engines"]:
        print(f"mkserve: unknown engine '{name}'; choose from {', '.join(entry['engines'])}", file=sys.stderr)
        return 2
    eng = entry["engines"][name]
    source = HERE / eng["source"]
    digest = source_digest(source)
    if not digest.startswith(eng["sha256"]):
        if not source.is_dir():
            print(f"mkserve: {source} does not match the registered sha256 ({digest[:16]} != {eng['sha256']})", file=sys.stderr)
            return 2
        # a source folder may have been edited on purpose: serve it, and say that it is not the registered engine
        print(f"mkserve: note: {source} differs from the registered source ({digest[:16]} != {eng['sha256']}); serving it as it is", file=sys.stderr)
    if shutil.which("nvcc") is None:
        print("mkserve: nvcc is not on PATH; the engine is built from CUDA source on this machine", file=sys.stderr)
        return 2

    work = Path(args.workdir or Path.home() / ".cache" / "mkserve") / f"{args.model.replace('/', '--')}_{card}_{name}_{digest[:12]}"
    if not (work / "engine" / "api.py").exists():
        work.mkdir(parents=True, exist_ok=True)
        if source.is_dir():      # a source tree in the repository: <source>/engine/...
            shutil.copytree(source, work, dirs_exist_ok=True, ignore=lambda d, names: [n for n in names if is_product(Path(n))])
        else:
            with tarfile.open(source) as tar:
                tar.extractall(work, filter="data")
    print(f"mkserve: engine {name} (track {eng['track']}, {eng['from']}), building in {work}", flush=True)
    build = work / "engine" / "build.sh"
    if build.exists():
        env = dict(os.environ, PATH=f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}")
        rc = subprocess.run(["bash", str(build)], cwd=work, env=env).returncode
        if rc:
            print(f"mkserve: the engine's build failed (exit {rc})", file=sys.stderr)
            return rc

    model_dir = args.model_dir
    if not model_dir:
        from huggingface_hub import snapshot_download
        print(f"mkserve: fetching {args.model} from Hugging Face (use --model-dir to serve a local snapshot)", flush=True)
        model_dir = snapshot_download(args.model)
    if not (Path(model_dir) / "tokenizer.json").exists():
        print(f"mkserve: {model_dir} has no tokenizer.json: not a model snapshot", file=sys.stderr)
        return 2

    sock = Path(tempfile.mkdtemp(prefix="mkserve-")) / "engine.sock"
    common = ["--model-dir", str(model_dir), "--sock", str(sock)]
    # An offline engine is driven step by step (its own request path batches only equal prompt lengths and was built
    # for the benchmark's short prompts); an online engine through its own submit/poll.
    mode = args.mode if args.mode != "auto" else ("native" if eng["kind"] == "online" else "step")
    worker = subprocess.Popen([sys.executable, "-m", "mkserve.server", "engine", "--workspace", str(work), "--mode", mode,
                               "--max-batch", str(args.max_batch or eng["max_batch"]), "--max-context", str(args.max_context or eng["max_context"]), *common],
                              cwd=HERE.parent, env=dict(os.environ, PYTHONPATH=f"{HERE.parent}:{os.environ.get('PYTHONPATH', '')}"))
    front = subprocess.Popen([sys.executable, "-m", "mkserve.server", "frontend", "--host", args.host, "--port", str(args.port), "--name", args.model,
                              "--engine-name", name, "--default-max-tokens", str(args.default_max_tokens), *(["--strict-sampling"] if args.strict_sampling else []), *common],
                             cwd=HERE.parent, env=dict(os.environ, PYTHONPATH=f"{HERE.parent}:{os.environ.get('PYTHONPATH', '')}"))

    def stop(*_):
        for p in (front, worker):
            if p.poll() is None:
                p.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    rc = 0
    while True:  # the server is the pair: when one process ends, the other is stopped
        for p in (worker, front):
            try:
                p.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                continue
            rc = p.returncode
            stop()
            for q in (front, worker):
                try:
                    q.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    q.kill()
            shutil.rmtree(sock.parent, ignore_errors=True)
            return 0 if rc in (0, -signal.SIGTERM, -signal.SIGINT) else rc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mkserve", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the models, cards and engines that can be served")
    s = sub.add_parser("serve", help="serve one model through its megakernel engine")
    s.add_argument("model", help="the model id, e.g. Qwen/Qwen3-8B")
    s.add_argument("--engine", default=None, help="an engine name from `list` (default: the model's default engine)")
    s.add_argument("--track", default=None, choices=["A", "B", "a", "b"], help="A: the general engines' number formats; B: lossless model-specific formats")
    s.add_argument("--card", default=None, help="registry key of the card (default: detected with nvidia-smi)")
    s.add_argument("--model-dir", default=None, help="a local Hugging Face snapshot of the model (default: download)")
    s.add_argument("--workdir", default=None, help="where engines are unpacked and built (default ~/.cache/mkserve)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--mode", choices=["auto", "native", "step"], default="auto", help="see mkserve/server.py: native = the engine's own batching (greedy); step = sampling and early stop, one batch at a time")
    s.add_argument("--max-batch", type=int, default=None)
    s.add_argument("--max-context", type=int, default=None)
    s.add_argument("--default-max-tokens", type=int, default=512)
    s.add_argument("--strict-sampling", action="store_true", help="refuse sampling requests in native mode instead of decoding greedily")
    args = parser.parse_args(argv)
    return cmd_list(args) if args.cmd == "list" else cmd_serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
