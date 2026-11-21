#!/usr/bin/env python
"""An OpenAI-compatible streaming /v1/completions over an engine under test, so that eval/client.py measures it
exactly as it measures vLLM and SGLang: over HTTP, one SSE chunk per token, text detokenized incrementally, on the
client's clock. Without this the engine is timed inside its own process and the comparison flatters it.

Two processes joined by a Unix socket, the shape the general engines have (an API front end and a model worker):

    serve_engine.py engine   --workspace WS --model-dir D --sock S [--max-batch 8] [--max-context 2048]
    serve_engine.py frontend --model-dir D --sock S --port P

`engine` runs in the evaluation venv (torch) with the working directory in WS and owns the GPU:
  * an engine with the online contract (submit / poll): requests are handed to submit() as they arrive and every
    poll() result is forwarded: admission, batching and prefill are the engine's;
  * an engine with only the offline contract (step): a cohort adapter. Requests that arrive within --gather-ms of
    the first one and have the same prompt length form a batch: reset(), the prompt through step(), then one
    step() + argmax per token with the chosen tokens copied to the host and sent at every step. Nothing is
    admitted while a cohort decodes, so this serves the steady workloads only (that is what such an engine is).
`frontend` (aiohttp, tokenizers) parses requests, detokenizes incrementally and writes the SSE stream.

Every request produces exactly max_tokens tokens (what `ignore_eos` asks of the other servers); sampling is greedy.
Prompts are token-id lists or text. Only streaming requests are served.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import queue
import socket
import sys
import threading
import time
from pathlib import Path


# ---------------------------------------------------------------------------------------------- the engine process

def engine_main(args) -> int:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _engine_runner import load_engine

    engine = load_engine(args.workspace, args.model_dir, args.max_batch, args.max_context)
    online = all(hasattr(engine, m) for m in ("submit", "poll"))
    Path(args.sock).unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.sock)
    server.listen(1)
    print(f"MK_SERVE engine ready: contract={'online' if online else 'offline (cohort adapter)'} "
          f"max_batch={args.max_batch} max_context={args.max_context} sock={args.sock}", flush=True)
    conn, _ = server.accept()
    inbox: queue.Queue = queue.Queue()

    def reader() -> None:
        with conn.makefile("r") as stream:
            for line in stream:
                inbox.put(json.loads(line))
        inbox.put(None)  # the front end is gone

    threading.Thread(target=reader, daemon=True).start()

    def send(events: list) -> None:
        conn.sendall((json.dumps(events, separators=(",", ":")) + "\n").encode())

    if online:
        while True:
            while True:
                try:
                    req = inbox.get_nowait()
                except queue.Empty:
                    break
                if req is None:
                    return 0
                engine.submit(req["rid"], req["ids"], req["n"])
            events = engine.poll()
            if events:
                send([[int(rid), [int(t) for t in tokens], bool(finished)] for rid, tokens, finished in events])
            else:
                time.sleep(0.0002)

    waiting: list[dict] = []
    with torch.no_grad():
        while True:
            if not waiting:
                req = inbox.get()
                if req is None:
                    return 0
                waiting.append(req)
            deadline = time.perf_counter() + args.gather_ms / 1e3
            while len(waiting) < args.max_batch and time.perf_counter() < deadline:
                try:
                    req = inbox.get(timeout=max(0.0, deadline - time.perf_counter()))
                except queue.Empty:
                    break
                if req is None:
                    return 0
                waiting.append(req)
            length = len(waiting[0]["ids"])
            cohort = [r for r in waiting if len(r["ids"]) == length][:args.max_batch]
            waiting = [r for r in waiting if all(r is not c for c in cohort)]
            ids = torch.tensor([r["ids"] for r in cohort], device="cuda")
            engine.reset()
            for t in range(length - 1):
                engine.step(ids[:, t])
            token = ids[:, -1]
            for k in range(max(r["n"] for r in cohort)):
                token = engine.step(token).argmax(-1)
                chosen = token.tolist()  # the host copy a streaming server needs at every step
                send([[r["rid"], [chosen[i]], k == r["n"] - 1] for i, r in enumerate(cohort) if k < r["n"]])


# ------------------------------------------------------------------------------------------- the front-end process

async def frontend_main(args) -> int:
    from aiohttp import web
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(args.model_dir) / "tokenizer.json"))
    for _ in range(int(args.wait_s * 10)):  # the engine process may still be loading
        if os.path.exists(args.sock):
            break
        await asyncio.sleep(0.1)
    reader, writer = await asyncio.open_unix_connection(args.sock, limit=1 << 24)
    queues: dict[int, asyncio.Queue] = {}
    counter = itertools.count(1)
    # Tokens per engine event (one poll() result for one request, or one step of the cohort adapter): for a
    # speculative engine that hands over each verification step's tokens at once, the acceptance length. GET
    # /mk/bursts (?reset=1) reports and optionally clears it; the evaluator reads it around each measurement.
    bursts = {"events": 0, "tokens": 0, "hist": {}}

    async def pump() -> None:
        while True:
            line = await reader.readline()
            if not line:
                os._exit(3)  # the engine process died: a server that answers nothing must not stay up
            for rid, tokens, finished in json.loads(line):
                q = queues.get(rid)
                if tokens:
                    bursts["events"] += 1
                    bursts["tokens"] += len(tokens)
                    bursts["hist"][len(tokens)] = bursts["hist"].get(len(tokens), 0) + 1
                if q is not None:
                    q.put_nowait((tokens, finished))

    async def burst_stats(request):
        report = {"events": bursts["events"], "tokens": bursts["tokens"],
                  "tokens_per_event": bursts["tokens"] / bursts["events"] if bursts["events"] else None,
                  "hist": {str(k): v for k, v in sorted(bursts["hist"].items())}}
        if request.query.get("reset"):
            bursts.update(events=0, tokens=0, hist={})
        return web.json_response(report)

    async def models(_request):
        return web.json_response({"object": "list", "data": [{"id": args.model, "object": "model"}]})

    async def completions(request):
        body = await request.json()
        if not body.get("stream"):
            return web.json_response({"error": {"message": "only streaming requests are served"}}, status=400)
        prompt = body.get("prompt")
        if isinstance(prompt, str):
            ids = tok.encode(prompt, add_special_tokens=False).ids
        elif isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt):
            ids = prompt
        else:
            return web.json_response({"error": {"message": "prompt must be text or a list of token ids"}}, status=400)
        n = int(body.get("max_tokens") or 16)
        with_ids = bool(body.get("mk_return_ids"))  # the evaluator's token check asks for the ids next to the text
        rid = next(counter)
        q: asyncio.Queue = asyncio.Queue()
        queues[rid] = q
        writer.write((json.dumps({"rid": rid, "ids": ids, "n": n}, separators=(",", ":")) + "\n").encode())
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(request)
        head = {"id": f"cmpl-{rid}", "object": "text_completion", "created": int(time.time()), "model": args.model}
        out: list[int] = []
        prefix = read = 0  # incremental detokenization as the general engines do it: text is released when it is whole
        try:
            done = False
            while not done:
                tokens, done = await q.get()
                for i, token in enumerate(tokens):
                    out.append(token)
                    last = done and i == len(tokens) - 1
                    before = tok.decode(out[prefix:read], skip_special_tokens=False)
                    now = tok.decode(out[prefix:], skip_special_tokens=False)
                    piece = ""
                    if len(now) > len(before) and (last or not now.endswith("�")):
                        piece, prefix, read = now[len(before):], read, len(out)
                    choice = {"index": 0, "text": piece, "logprobs": None, "finish_reason": "length" if last else None}
                    if with_ids:
                        choice["mk_token_ids"] = [token]
                    chunk = dict(head, choices=[choice])
                    await resp.write(b"data: " + json.dumps(chunk, separators=(",", ":")).encode() + b"\n\n")
            if (body.get("stream_options") or {}).get("include_usage"):
                usage = {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}
                await resp.write(b"data: " + json.dumps(dict(head, choices=[], usage=usage)).encode() + b"\n\n")
            await resp.write(b"data: [DONE]\n\n")
        finally:
            queues.pop(rid, None)
        return resp

    app = web.Application(client_max_size=1 << 26)
    app.add_routes([web.get("/v1/models", models), web.post("/v1/completions", completions), web.get("/mk/bursts", burst_stats)])
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", args.port).start()
    print(f"MK_SERVE frontend ready on port {args.port}", flush=True)
    await pump()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("role", choices=["engine", "frontend"])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--sock", required=True)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--gather-ms", type=float, default=50.0, help="offline engines: how long a cohort waits for more requests")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="m")
    parser.add_argument("--wait-s", type=float, default=900.0, help="frontend: how long to wait for the engine's socket")
    args = parser.parse_args()
    if args.role == "engine":
        return engine_main(args)
    return asyncio.run(frontend_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
