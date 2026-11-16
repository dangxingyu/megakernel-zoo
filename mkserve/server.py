#!/usr/bin/env python
"""The two processes behind `python -m mkserve serve`: an engine worker that owns the GPU and drives one megakernel
engine, and an HTTP front end with the OpenAI API. They are joined by a Unix socket (one JSON object per line), the
shape eval/serve_engine.py has; that file is the evaluator's measuring shim and is left alone, this one is for use.

    python -m mkserve.server engine   --workspace WS --model-dir D --sock S [--mode auto|native|step] ...
    python -m mkserve.server frontend --model-dir D --sock S --port P [--host H] --name MODEL ...

The engine worker has two ways to drive an engine (the contract is engine/api.py of the engine's source):

  native  the engine's own submit()/poll(): admission, batching and prefill are the engine's, requests of any
          length are taken while others decode. Greedy only, and the contract has no cancel: a request that ends
          early (end-of-sequence, a stop string, a client that left) is closed by the front end while its row
          decodes on to max_tokens inside the engine.
  step    reset()/step() driven from here: requests that arrive together and have prompts of the same length form
          a batch, the prompt is fed one token per step, then one step per token. Sampling (temperature, top_k,
          top_p, seed) and early stopping work, but nothing is admitted while a batch decodes and the first token is slow.

Front end: /v1/models, /health, /v1/completions and /v1/chat/completions, streaming or not. The chat template is
the model's own (tokenizer_config.json or chat_template.jinja), rendered with jinja2 as transformers does.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import itertools
import json
import os
import queue
import socket
import sys
import threading
import time
from pathlib import Path


# ---------------------------------------------------------------------------------------------- the engine worker

def load_engine(workspace: str, model_dir: str, max_batch: int, max_context: int):
    api = Path(workspace) / "engine" / "api.py"
    sys.path.insert(0, str(api.parent))
    spec = importlib.util.spec_from_file_location("engine_api", api)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Engine(model_dir, max_batch, max_context)


def sample_rows(torch, logits, temp, top_p, top_k, kmax, u):
    """One token per row from logits [R, V]: temperature, then top-k, then top-p on what is left (the order the
    general engines use), drawn by inverse CDF with one uniform number per row (u [R]), so a request's seed alone
    decides its tokens. temp, top_p [R] float; top_k [R] int, 0 = off; kmax = the largest top_k of the rows, or 0
    when a row has none (then the whole vocabulary is sorted). The k largest are taken before the temperature is
    applied (dividing by a positive number keeps the order), so only k values per row are touched after that."""
    if kmax:
        vals, idx = torch.topk(logits, kmax, dim=-1)           # sorted, largest first
    else:
        vals, idx = torch.sort(logits, dim=-1, descending=True)
    vals = vals.float() / temp[:, None]
    rank = torch.arange(vals.shape[-1], device=vals.device)[None, :]
    vals = vals.masked_fill((top_k[:, None] > 0) & (rank >= top_k[:, None]), float("-inf"))
    p = torch.softmax(vals, -1)
    p = p * (((p.cumsum(-1) - p) < top_p[:, None]))            # the smallest set whose mass reaches top_p
    cdf = (p / p.sum(-1, keepdim=True)).cumsum(-1)
    last = (p > 0).sum(-1, keepdim=True) - 1                   # never past the last token that was kept
    j = torch.minimum(torch.searchsorted(cdf, u[:, None]), last)
    return idx.gather(-1, j).squeeze(-1)


def engine_main(args) -> int:
    import numpy as np
    import torch

    os.chdir(args.workspace)
    engine = load_engine(args.workspace, args.model_dir, args.max_batch, args.max_context)
    has_native = all(hasattr(engine, m) for m in ("submit", "poll"))
    mode = args.mode if args.mode != "auto" else ("native" if has_native else "step")
    if mode == "native" and not has_native:
        print("MKSERVE this engine has no submit()/poll(): using step mode", flush=True)
        mode = "step"
    Path(args.sock).unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.sock)
    server.listen(1)
    print(f"MKSERVE engine ready: mode={mode} max_batch={args.max_batch} max_context={args.max_context}", flush=True)
    conn, _ = server.accept()
    conn.sendall((json.dumps({"hello": {"mode": mode, "max_batch": args.max_batch, "max_context": args.max_context}}) + "\n").encode())
    inbox: queue.Queue = queue.Queue()
    cancelled: set[int] = set()

    def reader() -> None:
        with conn.makefile("r") as stream:
            for line in stream:
                msg = json.loads(line)
                if "cancel" in msg:
                    cancelled.add(int(msg["cancel"]))
                else:
                    inbox.put(msg)
        inbox.put(None)  # the front end is gone

    threading.Thread(target=reader, daemon=True).start()

    def send(events: list) -> None:
        conn.sendall((json.dumps(events, separators=(",", ":")) + "\n").encode())

    if mode == "native":
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
            B = len(cohort)
            ids = torch.tensor([r["ids"] for r in cohort], device="cuda")
            temp = torch.tensor([float(r.get("temperature") or 0.0) for r in cohort], device="cuda")
            top_p = torch.tensor([float(r.get("top_p") or 1.0) for r in cohort], device="cuda")
            top_k = torch.tensor([int(r.get("top_k") or 0) for r in cohort], device="cuda")
            rows = (temp > 0).nonzero().flatten()              # the rows that sample; the others take the argmax
            n_max = max(r["n"] for r in cohort)
            if rows.numel():  # everything the sampler needs per step is prepared here: no host work inside the loop
                picked = rows.tolist()
                everyone = len(picked) == B
                ks = [int(cohort[i].get("top_k") or 0) for i in picked]
                kmax = max(ks) if all(k > 0 for k in ks) else 0
                # one uniform number per token and row, from the request's own seed
                uniform = torch.from_numpy(np.stack([np.random.default_rng(int(cohort[i]["seed"]) if cohort[i].get("seed") is not None else None)
                                                     .random(n_max, dtype=np.float32) for i in picked])).cuda()
                s_temp, s_top_p, s_top_k = temp[rows], top_p[rows], top_k[rows]
            eos = [set(r.get("eos") or ()) for r in cohort]
            done = [False] * B
            engine.reset()
            for t in range(length - 1):
                engine.step(ids[:, t])
            token = ids[:, -1]
            for k in range(n_max):
                logits = engine.step(token)
                if rows.numel() and s_temp.device != logits.device:  # a multi-card engine returns logits on a card of its choice
                    s_temp, s_top_p, s_top_k, uniform, rows = (x.to(logits.device) for x in (s_temp, s_top_p, s_top_k, uniform, rows))
                if rows.numel() and everyone:
                    token = sample_rows(torch, logits, s_temp, s_top_p, s_top_k, kmax, uniform[:, k])
                else:
                    token = logits.argmax(-1)
                    if rows.numel():
                        token[rows] = sample_rows(torch, logits[rows], s_temp, s_top_p, s_top_k, kmax, uniform[:, k])
                chosen = token.tolist()
                token = token.to(ids.device)
                events = []
                for i, r in enumerate(cohort):
                    if done[i]:
                        continue
                    last = k == r["n"] - 1 or chosen[i] in eos[i] or r["rid"] in cancelled
                    events.append([r["rid"], [chosen[i]], last])
                    done[i] = last
                if events:
                    send(events)
                if all(done):
                    break
            cancelled.difference_update(r["rid"] for r in cohort)


# ------------------------------------------------------------------------------------------------- the front end

def load_chat_template(model_dir: Path):
    """The model's chat template as a jinja2 template, with the names transformers gives it; None if it has none."""
    cfg_path = model_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    text = (model_dir / "chat_template.jinja").read_text() if (model_dir / "chat_template.jinja").exists() else cfg.get("chat_template")
    if isinstance(text, list):  # several named templates: take the default one
        text = next((t.get("template") for t in text if t.get("name") == "default"), None)
    if not text:
        return None, {}
    import datetime

    from jinja2.exceptions import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise TemplateError(message)

    from jinja2 import nodes
    from jinja2.ext import Extension, loopcontrols

    class Generation(Extension):
        """`{% generation %}...{% endgeneration %}` (transformers marks the assistant's own text with it): rendered as is."""
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
            return nodes.Scope(body, lineno=lineno)

    # the extensions transformers compiles chat templates with: `break` / `continue` and the generation marker
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols, Generation])
    env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, **kw: json.dumps(x, ensure_ascii=ensure_ascii, indent=indent)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    special = {}
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        value = cfg.get(key)
        special[key] = value.get("content") if isinstance(value, dict) else (value or "")
    return env.from_string(text), special


END_OF_TURN = ("<|im_end|>", "<|eot_id|>", "<|end_of_turn|>", "<end_of_turn>", "<|endoftext|>", "<|content_model_end_sampling|>", "<|END_OF_TURN_TOKEN|>")


def eos_ids(model_dir: Path, tok) -> list[int]:
    ids: set[int] = set()
    gen = model_dir / "generation_config.json"
    if gen.exists():
        value = json.loads(gen.read_text()).get("eos_token_id")
        ids.update(value if isinstance(value, list) else [value] if value is not None else [])
    cfg_path = model_dir / "tokenizer_config.json"
    if cfg_path.exists():
        value = json.loads(cfg_path.read_text()).get("eos_token")
        value = value.get("content") if isinstance(value, dict) else value
        if value and tok.token_to_id(value) is not None:
            ids.add(tok.token_to_id(value))
    model_cfg = model_dir / "config.json"
    if model_cfg.exists():   # some checkpoints name the end token only here
        value = json.loads(model_cfg.read_text()).get("eos_token_id")
        ids.update(value if isinstance(value, list) else [value] if value is not None else [])
    for name in END_OF_TURN:  # the token a chat template ends the assistant's turn with, when the vocabulary has it
        if tok.token_to_id(name) is not None:
            ids.add(tok.token_to_id(name))
    return sorted(int(i) for i in ids)


async def frontend_main(args) -> int:
    from aiohttp import web
    from tokenizers import Tokenizer

    model_dir = Path(args.model_dir)
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    template, special = load_chat_template(model_dir)
    eos = eos_ids(model_dir, tok)
    vocab = tok.get_vocab_size(with_added_tokens=True)   # ids a request may carry: the model's embedding rows when its config says how many
    model_cfg = model_dir / "config.json"
    if model_cfg.exists():
        vocab = int(json.loads(model_cfg.read_text()).get("vocab_size") or vocab)
    for _ in range(int(args.wait_s * 10)):  # the engine worker may still be loading
        if os.path.exists(args.sock):
            break
        await asyncio.sleep(0.1)
    reader, writer = await asyncio.open_unix_connection(args.sock, limit=1 << 24)
    hello = json.loads(await reader.readline())["hello"]
    native = hello["mode"] == "native"
    queues: dict[int, asyncio.Queue] = {}
    counter = itertools.count(1)
    warned = {"sampling": False}

    async def pump() -> None:
        while True:
            line = await reader.readline()
            if not line:
                os._exit(3)  # the engine worker died: a server that answers nothing must not stay up
            for rid, tokens, finished in json.loads(line):
                q = queues.get(rid)
                if q is not None:
                    q.put_nowait((tokens, finished))

    def error(message: str, status: int = 400):
        return web.json_response({"error": {"message": message, "type": "invalid_request_error"}}, status=status)

    async def models(_request):
        return web.json_response({"object": "list", "data": [{"id": args.name, "object": "model", "owned_by": "mkserve",
                                                                 "engine": args.engine_name, "mode": hello["mode"]}]})

    async def health(_request):
        return web.json_response({"status": "ok", "engine": args.engine_name, "mode": hello["mode"], "max_batch": hello["max_batch"],
                                  "max_context": hello["max_context"]})

    async def generate(request, body, ids: list[int], chat: bool):
        """Run one request through the engine and answer in the completions or the chat format."""
        stream = bool(body.get("stream"))
        n = body.get("max_completion_tokens") or body.get("max_tokens") or args.default_max_tokens
        n = int(n)
        # The engines do not check their inputs (an out-of-range token id is an out-of-bounds read on the card): every limit is enforced here.
        if not ids:
            return error("the prompt is empty")
        if n < 1:
            return error("max_tokens must be at least 1")
        if any(t < 0 or t >= vocab for t in ids):
            return error(f"token ids must be in [0, {vocab})")
        if len(ids) + n > hello["max_context"]:
            return error(f"prompt ({len(ids)} tokens) plus max_tokens ({n}) exceeds this engine's context of {hello['max_context']} tokens")
        temperature = body.get("temperature")
        sampling = temperature is not None and float(temperature) > 0
        if sampling and native:
            if args.strict_sampling:
                return error("this engine decodes greedily in native mode; send temperature 0, or start the server with --mode step")
            if not warned["sampling"]:
                warned["sampling"] = True
                print("MKSERVE warning: sampling parameters are ignored in native mode (greedy decoding); use --mode step for sampling", flush=True)
            sampling = False
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else [s for s in stops if s]
        hold = max((len(s) for s in stops), default=1) - 1  # text that could still turn into a stop string is held back
        stop_ids = set() if body.get("ignore_eos") else set(eos)
        rid = next(counter)
        q: asyncio.Queue = asyncio.Queue()
        queues[rid] = q
        msg = {"rid": rid, "ids": ids, "n": n, "eos": sorted(stop_ids)}
        if sampling:
            msg.update(temperature=float(temperature), top_p=float(body.get("top_p") or 1.0), top_k=int(body.get("top_k") or 0), seed=body.get("seed"))
        writer.write((json.dumps(msg, separators=(",", ":")) + "\n").encode())
        created, kind = int(time.time()), ("chat.completion" if chat else "text_completion")
        head = {"id": f"{'chatcmpl' if chat else 'cmpl'}-{rid}", "object": kind + (".chunk" if chat and stream else ""), "created": created, "model": args.name}

        def chunk(text: str, finish, first: bool = False) -> bytes:
            if chat:
                delta = {"role": "assistant", "content": text} if first else ({"content": text} if text else {})
                choice = {"index": 0, "delta": delta, "finish_reason": finish}
            else:
                choice = {"index": 0, "text": text, "logprobs": None, "finish_reason": finish}
            return b"data: " + json.dumps(dict(head, choices=[choice]), separators=(",", ":"), ensure_ascii=False).encode() + b"\n\n"

        resp = None
        if stream:
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await resp.prepare(request)
        out: list[int] = []
        text, sent, finish = "", 0, None
        prefix = read = 0  # incremental detokenization: text is released when it is whole
        first = True
        try:
            done = False
            while not done and finish is None:
                tokens, done = await q.get()
                for token in tokens:
                    if token in stop_ids:
                        finish = "stop"
                        break
                    out.append(token)
                    before = tok.decode(out[prefix:read], skip_special_tokens=True)
                    now = tok.decode(out[prefix:], skip_special_tokens=True)
                    if len(now) > len(before) and not now.endswith("�"):
                        text += now[len(before):]
                        prefix, read = read, len(out)
                    cut = min((text.find(s) for s in stops if s in text), default=-1)
                    if cut >= 0:
                        text, finish = text[:cut], "stop"
                        break
                    if len(out) >= n:
                        finish = "length"
                        break
                    if stream:  # one chunk per token (its text may be empty), so that a client can time tokens
                        upto = max(sent, len(text) - hold)
                        await resp.write(chunk(text[sent:upto], None, first))
                        sent, first = upto, False
                if done and finish is None:
                    finish = "length" if len(out) >= n else "stop"
            if finish == "length":  # nothing follows: release text that was waiting for more bytes
                tail = tok.decode(out[prefix:], skip_special_tokens=True)[len(tok.decode(out[prefix:read], skip_special_tokens=True)):]
                text += tail.rstrip("�")
            usage = {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}
            if not stream:
                choice = ({"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish} if chat
                          else {"index": 0, "text": text, "logprobs": None, "finish_reason": finish})
                return web.json_response(dict(head, choices=[choice], usage=usage), dumps=lambda o: json.dumps(o, ensure_ascii=False))
            await resp.write(chunk(text[sent:], finish, first))
            if (body.get("stream_options") or {}).get("include_usage"):
                await resp.write(b"data: " + json.dumps(dict(head, choices=[], usage=usage)).encode() + b"\n\n")
            await resp.write(b"data: [DONE]\n\n")
            return resp
        finally:
            queues.pop(rid, None)
            if not done:  # ended early: a step-mode batch can drop the row; a native engine decodes it to the end
                writer.write((json.dumps({"cancel": rid}) + "\n").encode())

    async def completions(request):
        body = await request.json()
        prompt = body.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str):
            prompt = prompt[0]
        if isinstance(prompt, str):
            ids = tok.encode(prompt, add_special_tokens=False).ids
        elif isinstance(prompt, list) and prompt and all(isinstance(t, int) for t in prompt):
            ids = prompt
        else:
            return error("prompt must be one string or one list of token ids")
        return await generate(request, body, ids, chat=False)

    async def chat_completions(request):
        body = await request.json()
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return error("messages must be a non-empty list")
        if template is None:
            return error("this model has no chat template; use /v1/completions")
        for m in messages:  # content given as a list of text parts becomes one string
            if isinstance(m.get("content"), list):
                m["content"] = "".join(p.get("text", "") for p in m["content"] if isinstance(p, dict) and p.get("type") == "text")
        try:
            text = template.render(messages=messages, tools=body.get("tools"), add_generation_prompt=True, **special,
                                   **(body.get("chat_template_kwargs") or {}))
        except Exception as exc:  # the template rejects the conversation (for example an unknown role)
            return error(f"chat template: {exc}")
        return await generate(request, body, tok.encode(text, add_special_tokens=False).ids, chat=True)

    app = web.Application(client_max_size=1 << 26)
    app.add_routes([web.get("/v1/models", models), web.get("/health", health),
                    web.post("/v1/completions", completions), web.post("/v1/chat/completions", chat_completions)])
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, args.host, args.port).start()
    print(f"MKSERVE serving {args.name} on http://{args.host}:{args.port}/v1 (engine {args.engine_name}, {hello['mode']} mode)", flush=True)
    await pump()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("role", choices=["engine", "frontend"])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--sock", required=True)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--mode", choices=["auto", "native", "step"], default="auto")
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--gather-ms", type=float, default=20.0, help="step mode: how long a batch waits for more requests")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--name", default="model", help="the model id the API reports")
    parser.add_argument("--engine-name", default="")
    parser.add_argument("--default-max-tokens", type=int, default=512)
    parser.add_argument("--strict-sampling", action="store_true", help="refuse sampling requests in native mode instead of decoding greedily")
    parser.add_argument("--wait-s", type=float, default=1800.0, help="front end: how long to wait for the engine worker's socket")
    args = parser.parse_args(argv)
    if args.role == "engine":
        return engine_main(args)
    return asyncio.run(frontend_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
