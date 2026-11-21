#!/usr/bin/env python3
"""Check an engine directory before opening a pull request. Static checks only: no GPU, no build.

    python scripts/check_engine.py engines/<model>_<gpu>/<name>        # one engine
    python scripts/check_engine.py --all                               # every engine in engines/

What it checks:
  layout     engine/api.py, engine/build.sh and engine/TRACK exist; no build products or weights are committed
  TRACK      the first word is A, B or C; later words name track-B storage options ("B W12U L0T", "A W12"), which make
             a track-A engine a track-B engine; C is exact speculative decoding (draft and verify; its tokens equal its
             own step-by-step greedy tokens), whatever options follow
  contract   engine/api.py defines class Engine(model_dir, max_batch, max_context) with reset() and step()
             (offline engines) or submit() and poll() (online engines, the name starts with "online")
  timed path engine/api.py imports no general inference engine (vllm, sglang, ...) and no transformers model code
             (transformers configs, tokenizers and rotary-embedding tables may be used at load time)
  registry   mkserve/registry.json has an entry whose "source" is this directory, with the required fields, a track
             that agrees with engine/TRACK, and a sha256 that matches the directory (mkserve's source_digest)

It does not replace the evaluator: an engine is accepted only with an eval/kernel_eval.py result of "valid" on the GPU
it was written for (see CONTRIBUTING.md).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mkserve.__main__ import is_product, source_digest  # noqa: E402  (the digest the server checks)

FORBIDDEN_IMPORTS = ("vllm", "sglang", "tensorrt_llm", "lmdeploy", "exllamav2", "llama_cpp", "mlc_llm")
# transformers is allowed at load time for configuration and tables, never for a model or its layers
TRANSFORMERS_OK = re.compile(r"^(AutoConfig|AutoTokenizer|PretrainedConfig|\w*Config|\w*RotaryEmbedding)$")
REGISTRY_FIELDS = ("track", "kind", "source", "sha256", "from", "max_batch", "max_context", "what")
BIG_FILE = 5 * 1024 * 1024
NAME = re.compile(r"^(offline|online)-[a-z0-9][a-z0-9-]*$")
DIR = re.compile(r"^[a-z0-9][a-z0-9.\-]*_[a-z0-9]+(x\d+)?$")


def engine_class(tree: ast.Module):
    """class Engine, or the class that `Engine = X` names anywhere in the module; "imported" if Engine is only imported."""
    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    if "Engine" in classes:
        return classes["Engine"]
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "Engine" for t in n.targets) \
                and isinstance(n.value, ast.Name) and n.value.id in classes:
            return classes[n.value.id]
    if any(isinstance(n, ast.ImportFrom) and any(a.name == "Engine" for a in n.names) for n in ast.walk(tree)):
        return "imported"
    return None


def check(path: Path, registry: dict) -> list[tuple[bool, str]]:
    out: list[tuple[bool, str]] = []
    ok = lambda cond, msg: out.append((bool(cond), msg))
    rel = path.resolve().relative_to(ROOT).as_posix() if path.resolve().is_relative_to(ROOT) else str(path)
    eng = path / "engine"
    ok(DIR.match(path.parent.name), f"directory name '{path.parent.name}' is <model>_<gpu> (lower case, e.g. qwen3-8b_h100, inkling-2-small_b200x4)")
    ok(NAME.match(path.name), f"engine name '{path.name}' is offline-<x> or online-<x> (e.g. offline-a, online-b)")
    for f in ("api.py", "build.sh", "TRACK"):
        ok((eng / f).is_file(), f"engine/{f} exists")
    products = [p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() and is_product(p.relative_to(path))]
    ok(not products, "no build products (.so, .o, .pyc, build/, __pycache__/)" + (f": {products[:4]}" if products else ""))
    big = [p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() and p.stat().st_size > BIG_FILE]
    ok(not big, "no file over 5 MB (weights and tables are made at load time)" + (f": {big[:4]}" if big else ""))

    track = None
    if (eng / "TRACK").is_file():
        words = (eng / "TRACK").read_text().split()
        first = words[0].upper() if words else None
        ok(first in ("A", "B", "C"), f"engine/TRACK starts with A, B or C (found {' '.join(words) or 'nothing'})")
        # option words after the first one (W12, W12U, L0T, ...) are track-B storage formats: "A W12" is the one-token
        # data flow with 12-bit weights, which the registry reports as track B; track C (speculative decoding) stays C
        track = first if first == "C" else "B" if first == "B" or len(words) > 1 else first

    online = path.name.startswith("online")
    if (eng / "api.py").is_file():
        src = (eng / "api.py").read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            ok(False, f"engine/api.py parses ({e})")
            tree = None
        if tree is not None:
            cls = engine_class(tree)
            ok(cls is not None, "engine/api.py defines class Engine (or binds the name Engine to a class)")
            if isinstance(cls, ast.ClassDef):
                methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
                init = methods.get("__init__")
                params = [a.arg for a in init.args.args[1:]] if init else []
                ok(params[:3] == ["model_dir", "max_batch", "max_context"], f"Engine.__init__(model_dir, max_batch, max_context) (found {params})")
                need = ("submit", "poll", "idle") if online else ("reset", "step")
                missing = [m for m in need if m not in methods]
                ok(not missing, f"Engine has {', '.join(need)}" + (f" (missing {', '.join(missing)})" if missing else ""))
            mods, tf_bad = set(), []
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    mods |= {a.name.split(".")[0] for a in n.names}
                    tf_bad += [a.name for a in n.names if a.name.split(".")[0] == "transformers"]
                elif isinstance(n, ast.ImportFrom) and n.module:
                    mods.add(n.module.split(".")[0])
                    if n.module.split(".")[0] == "transformers":
                        tf_bad += [f"transformers.{a.name}" for a in n.names if not TRANSFORMERS_OK.match(a.name)]
            bad = sorted(mods & set(FORBIDDEN_IMPORTS)) + sorted(set(tf_bad))
            ok(not bad, "no general inference engine or library model code imported in engine/api.py" + (f": {bad}" if bad else ""))

    entries = [(m, g, n, e) for m, gpus in registry.get("models", {}).items() for g, ge in gpus.items()
               for n, e in ge.get("engines", {}).items() if (ROOT / "mkserve" / e.get("source", "")).resolve() == path.resolve()]
    ok(entries, f"mkserve/registry.json has an entry with source ../{rel}")
    for model, gpu, name, e in entries:
        missing = [f for f in REGISTRY_FIELDS if f not in e]
        ok(not missing, f"registry entry {model} / {gpu} / {name} has {', '.join(REGISTRY_FIELDS)}" + (f" (missing {', '.join(missing)})" if missing else ""))
        ok(name == path.name, f"registry name '{name}' equals the directory name '{path.name}'")
        if track:
            ok(e.get("track") == track, f"registry track {e.get('track')} agrees with engine/TRACK ({track})")
        ok(e.get("kind") == ("online" if online else "offline"), f"registry kind '{e.get('kind')}' agrees with the name")
        digest = source_digest(path)
        ok(digest.startswith(str(e.get("sha256", "-"))), f"registry sha256 {e.get('sha256')} matches the directory ({digest[:16]})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("engine", nargs="?", help="engines/<model>_<gpu>/<name>")
    ap.add_argument("--all", action="store_true", help="check every engine in engines/")
    args = ap.parse_args()
    if not args.engine and not args.all:
        ap.error("give an engine directory or --all")
    registry = json.loads((ROOT / "mkserve" / "registry.json").read_text())
    paths = sorted(p for p in (ROOT / "engines").glob("*/*") if p.is_dir()) if args.all else [Path(args.engine)]
    failed = 0
    for path in paths:
        results = check(path, registry)
        bad = [m for good, m in results if not good]
        failed += bool(bad)
        print(f"{'ok  ' if not bad else 'FAIL'} {path}")
        for good, msg in results:
            if not good or not args.all:
                print(f"     {'✓' if good else '✗'} {msg}")
    if args.all:
        print(f"{len(paths) - failed} of {len(paths)} engines pass")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
