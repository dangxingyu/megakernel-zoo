"""`mkserve doctor`, `mkserve bench`, `mkserve chat`: the small tools around the server."""
from __future__ import annotations

import asyncio
import importlib
import json
import random
import shutil
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- doctor
def doctor(args) -> int:
    from mkserve.__main__ import detect_card
    ok = True

    def line(good: bool, what: str, fix: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"  {'ok ' if good else 'NO '} {what}" + ("" if good or not fix else f"\n        fix: {fix}"))

    print("mkserve doctor")
    from mkserve.__main__ import memory_gb, resolve_card
    card, name = detect_card()
    mac = sys.platform == "darwin"
    if mac:  # Apple silicon: MLX and run-time Metal, no CUDA toolkit and no torch
        mem = memory_gb()
        line(card is not None, f"chip: {name}" + (f", {mem:.0f} GB unified memory" if mem else ""), "an Apple-silicon Mac is needed")
        runtime = ("mlx", "the Mac engines run on MLX and Metal")
    else:
        line(card is not None, f"card: {name}" + (f" (engines are written for: {card})" if card else ""),
             "an NVIDIA H100 or B200 is needed; engines are compiled for one card type")
        nvcc = shutil.which("nvcc")
        ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1] if nvcc else ""
        line(bool(nvcc), f"CUDA toolkit: {ver or 'nvcc not on PATH'}", "install the CUDA toolkit (12.8 or newer) and put its bin/ on PATH")
        runtime = ("torch", "the engines run under PyTorch")
    for mod, why in (runtime, ("numpy", ""), ("aiohttp", "the HTTP front end"),
                     ("tokenizers", "detokenization"), ("jinja2", "chat templates"), ("huggingface_hub", "downloading weights")):
        try:
            m = importlib.import_module(mod)
            extra = ""
            if mod == "torch":
                extra = f" (CUDA {m.version.cuda}, device {'found' if m.cuda.is_available() else 'NOT visible'})"
                if not m.cuda.is_available():
                    line(False, f"torch {m.__version__}{extra}", "a CUDA build of torch is needed: pip install torch --index-url https://download.pytorch.org/whl/cu128")
                    continue
            if mod == "mlx":
                import mlx.core as mxc
                v = tuple(int(x) for x in mxc.__version__.split(".")[:2])
                line(v >= (0, 32), f"mlx {mxc.__version__}", "pip install -U 'mlx>=0.32' (NVFP4 / MXFP8 matmuls and masked fused attention)")
                continue
            line(True, f"{mod} {getattr(m, '__version__', '')}{extra}")
        except ImportError:
            line(False, f"{mod} missing ({why})" if why else f"{mod} missing", f"pip install {mod.replace('_', '-')}")
    reg = json.loads((HERE / "registry.json").read_text())
    here = [(m, n) for m, cards in reg["models"].items() for key in [resolve_card(cards, card)[0]] if key for n in cards[key]["engines"]]
    line(bool(here), f"engines for this card: {len(here)} ({', '.join(sorted({m for m, _ in here})) or 'none'})",
         "see `mkserve list` for the (model, card) pairs that have engines")
    if args.model:
        cards = reg["models"].get(args.model, {})
        key, note = resolve_card(cards, card)
        line(key is not None, f"{args.model} on this card: {'engine ' + cards[key]['default'] if key else 'no engine'}" + (f" ({note})" if note else ""))
        need = cards[key].get("min_memory_gb") if key else None
        if need and mac and memory_gb():
            line(memory_gb() >= need, f"memory for {args.model}: about {need} GB needed, {memory_gb():.0f} GB here",
                 "a Mac with more unified memory is needed (the weights alone are larger than what is free here)")
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(args.model, "config.json")
            print(f"  --  weights of {args.model}: " + ("in the Hugging Face cache" if isinstance(cached, str) else
                  f"not in the Hugging Face cache (`mkserve pull {args.model}` downloads them; `--model-dir` serves a local copy)"))
        except ImportError:
            pass
    print("ready" if ok else "not ready: fix the lines marked NO")
    return 0 if ok else 1


# ---------------------------------------------------------------- http helpers
def _headers(key):
    return {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}


async def _model_id(session, url, key):
    async with session.get(f"{url}/v1/models", headers=_headers(key)) as r:
        return (await r.json())["data"][0]["id"]


async def _one(session, url, key, model, prompt, max_tokens):
    """One streamed completion: (send time, first token time, last token time, tokens)."""
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "ignore_eos": True, "stream_options": {"include_usage": True}}
    t0, first, last, chunks, usage = time.perf_counter(), None, None, 0, None
    async with session.post(f"{url}/v1/completions", json=body, headers=_headers(key)) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:300]}")
        async for raw in r.content:
            line = raw.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            msg = json.loads(line[5:])
            if msg.get("usage"):
                usage = msg["usage"]
            if msg.get("choices"):
                now = time.perf_counter()
                first = first or now
                last = now
                chunks += 1
    tokens = (usage or {}).get("completion_tokens") or chunks
    return t0, first, last, tokens


async def _bench(args) -> int:
    import aiohttp
    url = args.url.rstrip("/")
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        model = args.model or await _model_id(session, url, args.api_key)
        rng = random.Random(0)
        print(f"mkserve bench: {url}, model {model}, prompt {args.prompt_tokens} random token ids, {args.max_tokens} new tokens, "
              f"greedy, end of sequence ignored, {args.rounds} rounds after a warm-up")
        await asyncio.gather(*[_one(session, url, args.api_key, model, [rng.randrange(1000, 30000) for _ in range(args.prompt_tokens)], 16)
                               for _ in range(max(args.batch))])
        print(f"{'batch':>5s} {'ms per step':>12s} {'tokens/s':>10s} {'TTFT ms':>9s}   (median over rounds; step = one token of every sequence)")
        for b in args.batch:
            steps, ttfts = [], []
            for _ in range(args.rounds):
                prompts = [[rng.randrange(1000, 30000) for _ in range(args.prompt_tokens)] for _ in range(b)]
                res = await asyncio.gather(*[_one(session, url, args.api_key, model, p, args.max_tokens) for p in prompts])
                per_req = [(last - first) / max(1, n - 1) for _, first, last, n in res if first and last and n > 1]
                steps.append(st.median(per_req) * 1e3)
                ttfts += [(first - t0) * 1e3 for t0, first, _, _ in res if first]
            ms = st.median(steps)
            print(f"{b:5d} {ms:12.2f} {b * 1e3 / ms:10.0f} {st.median(ttfts):9.0f}")
    return 0


def bench(args) -> int:
    return asyncio.run(_bench(args))


# ---------------------------------------------------------------- chat
async def _chat(args) -> int:
    import aiohttp
    url = args.url.rstrip("/")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as session:
        model = args.model or await _model_id(session, url, args.api_key)
        print(f"chatting with {model} at {url} (empty line or /exit to quit, /reset to forget the conversation)")
        history = []
        while True:
            try:
                user = input("\n> ").strip()
            except EOFError:
                return 0
            if user in ("", "/exit", "/quit"):
                return 0
            if user == "/reset":
                history = []
                continue
            history.append({"role": "user", "content": user})
            body = {"model": model, "messages": history, "max_tokens": args.max_tokens, "temperature": args.temperature,
                    "stream": True, "stream_options": {"include_usage": True}}
            text, t0, first, n = "", time.perf_counter(), None, 0
            async with session.post(f"{url}/v1/chat/completions", json=body, headers=_headers(args.api_key)) as r:
                if r.status != 200:
                    print(f"error {r.status}: {(await r.text())[:300]}")
                    history.pop()
                    continue
                async for raw in r.content:
                    line = raw.strip()
                    if not line.startswith(b"data:") or line == b"data: [DONE]":
                        continue
                    msg = json.loads(line[5:])
                    if msg.get("usage"):
                        n = msg["usage"].get("completion_tokens", n)
                    for ch in msg.get("choices") or []:
                        piece = (ch.get("delta") or {}).get("content") or ""
                        if piece:
                            first = first or time.perf_counter()
                            text += piece
                            print(piece, end="", flush=True)
            dt = time.perf_counter() - (first or t0)
            print(f"\n  [{n} tokens, {n / dt:.0f} tokens/s, first token after {((first or t0) - t0) * 1e3:.0f} ms]" if n and dt > 0 else "")
            history.append({"role": "assistant", "content": text})


def chat(args) -> int:
    try:
        return asyncio.run(_chat(args))
    except KeyboardInterrupt:
        return 0
