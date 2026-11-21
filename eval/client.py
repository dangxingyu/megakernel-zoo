#!/usr/bin/env python
"""Workload driver for any server with an OpenAI-compatible streaming /v1/completions.

    python eval/client.py steady --base http://127.0.0.1:8000 --model m1 --model-dir DIR \
        --batch 8 --prompt-tokens 64 --max-tokens 1024 --repeats 5 --out results.json

All timing is done here, on the client, with a monotonic clock: nothing a server reports about its own
speed is used. Every number is end-to-end (HTTP and streaming included) unless its name says otherwise.

steady (W1, W2): `batch` streams start together and each asks for `max_tokens` new tokens. Reported per repeat:
  e2e_tok_s      all completion tokens / (last token time - first send time)             [end-to-end]
  decode_tok_s   tokens that arrive while every stream is decoding / that window         [end-to-end, decode phase]
  ttft_s         per stream, send to first token
  tpot_ms        gaps between consecutive tokens of one stream (p50, p99, max)
A stream that ends early (EOS before max_tokens), or whose chunk count differs from the completion tokens the
server reports, is flagged: `clean` is false and the repeat should not be used.

arrivals (W3): `streams` background streams decode without pause (each is a loop of `max_tokens`-token requests on
short prompts). After `warmup_s`, a long prompt nobody has seen arrives every `period_s`, asking for
`arrival_max_tokens` tokens. Before the background starts, prompts of the same length are sent alone.
  decode_kept    background tok/s inside prefill windows (arrival sent -> its first token) / outside them
  ttft_ratio     arrival TTFT under load / median TTFT of the same length alone
  gap_ms         gaps between consecutive background tokens, overall and inside prefill windows

churn (context switching): `batch` clients, each sending one short request after another (closed loop). A request
is `prefix_tokens` shared by all requests (a prefix cache can reuse them) plus `prompt_tokens` of its own, and
asks for `max_tokens` new tokens, so every few steps a sequence leaves the batch and a new one enters: the
engine's admission, prefix lookup, KV allocation and release run constantly. Measured over `duration_s` after
`warmup_s`: requests/s, output tok/s (to compare with steady decode at the same batch), TTFT, token gaps, and
how many prompt tokens the server says came from its cache.

replay (authentic arrivals): replays a production trace whose records are {timestamp ms, input_length,
output_length, hash_ids} (the Mooncake FAST'25 traces from Kimi: hash_ids are prefix hashes of 512-token blocks).
Prompts are synthesized so that records sharing leading hashes share exactly those leading tokens. Requests are
sent open loop at the trace's own arrival times, compressed by `rate_scale` (2 = twice the trace's rate), for
`duration_s` of scaled time. Inputs longer than `max_input_tokens` are cut at the end (the shared prefix is kept).
Reported: TTFT and token-gap percentiles, output tok/s, the share of requests within the SLOs, cached prompt
tokens as the server reports them, and how many requests were still unfinished when the replay stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts  # noqa: E402


class Nvml:
    """GPU utilization as NVML reports it (share of its sample period with a kernel running), polled in a
    thread for every card NVML sees. mean(a, b) averages the distinct samples taken between two perf_counter times:
    `mean` is card 0's (the one-card figure, as before), `per_card` lists every card's, `cards_busy` counts the cards
    whose mean is above 5% (an engine on one card of a node and one on all of them are told apart)."""

    def __init__(self, period: float = 0.05) -> None:
        import threading

        import pynvml
        pynvml.nvmlInit()
        self._nvml = pynvml
        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        self.samples: list[tuple[float, list[int]]] = []
        self._stop = False
        self._period = period
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            self.samples.append((time.perf_counter(), [self._nvml.nvmlDeviceGetUtilizationRates(h).gpu for h in self._handles]))
            time.sleep(self._period)

    def mean(self, a: float, b: float) -> dict | None:
        # NVML's own sample period is 1/6 s or longer: skip the first 0.5 s of the window so that every sample
        # used describes time inside it.
        rows = [u for t, u in self.samples if a + 0.5 <= t <= b]
        if not rows:
            return None
        per_card = [statistics.fmean(col) for col in zip(*rows)]
        values = [r[0] for r in rows]
        return {"mean": statistics.fmean(values), "min": min(values), "n": len(values),
                "per_card": [round(x, 1) for x in per_card], "cards_busy": sum(x > 5.0 for x in per_card)}


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else float("nan")


def has_token(choice: dict) -> bool:
    """Whether a stream chunk carries generated output. Completions chunks have `text`; chat chunks (and MPK's
    completions endpoint) have `delta` with content or reasoning. An empty string still counts while the stream
    is not finished: detokenizers hold back incomplete characters."""
    if "delta" in choice:
        delta = choice["delta"] or {}
        values = [delta.get(k) for k in ("content", "reasoning_content", "reasoning")]
        return any(values) or (any(v is not None for v in values) and choice.get("finish_reason") is None)
    text = choice.get("text")
    return bool(text) or (text is not None and choice.get("finish_reason") is None)


def chunk_token_count(choice: dict) -> int | None:
    """How many tokens a stream chunk carries, when the server says: our front end's mk_token_ids, or the per-token
    entries of OpenAI logprobs (completions: logprobs.tokens; chat: logprobs.content). None when it does not say."""
    if isinstance(choice.get("mk_token_ids"), list):
        return len(choice["mk_token_ids"])
    lp = choice.get("logprobs")
    if isinstance(lp, dict):
        if isinstance(lp.get("tokens"), list):
            return len(lp["tokens"])
        if isinstance(lp.get("content"), list):
            return len(lp["content"])
    return None


def chunk_weights(record: dict, tokens: int) -> list[float]:
    """Tokens per content chunk of one stream (track C, token-based rates). The counts the server reported when it
    reported them for every chunk and they add up to the stream's completion tokens; otherwise the first chunk
    (the prompt's output) counts 1 and the stream's other tokens are spread evenly over its other chunks. A server
    that sends one chunk per token gets weight 1 for every chunk: the token-based rate is then exactly the chunk rate."""
    n = len(record["chunks"])
    known = record.get("chunk_tokens") or []
    if n and len(known) == n and all(k is not None for k in known) and sum(known) == tokens:
        return [float(k) for k in known]
    if n <= 1:
        return [float(tokens)] * n
    rest = (tokens - 1) / (n - 1)
    return [1.0] + [rest] * (n - 1)


async def one_stream(session: aiohttp.ClientSession, base: str, body: dict) -> dict:
    """Send one streaming completion. Returns send time, the arrival time of every content chunk, usage."""
    path = "/v1/chat/completions" if "messages" in body else "/v1/completions"
    record = {"t_send": time.perf_counter(), "chunks": [], "chunk_tokens": [], "usage": None, "finish": None, "error": None,
              "text_len": 0, "text": []}
    try:
        async with session.post(base + path, json=body) as resp:
            if resp.status != 200:
                record["error"] = f"HTTP {resp.status}: {(await resp.text())[:300]}"
                return record
            async for raw in resp.content:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                now = time.perf_counter()
                event = json.loads(data)
                if event.get("error"):
                    record["error"] = f"stream error: {str(event['error'])[:300]}"
                for choice in event.get("choices") or []:
                    if has_token(choice):
                        record["chunks"].append(now)
                        record["chunk_tokens"].append(chunk_token_count(choice))
                    piece = choice.get("text") if "delta" not in choice else (choice.get("delta") or {}).get("content")
                    if piece:
                        record["text"].append(piece)
                    if choice.get("finish_reason"):
                        record["finish"] = choice["finish_reason"]
                if event.get("usage"):
                    record["usage"] = event["usage"]
    except Exception as error:  # a broken stream is a result, not a crash
        record["error"] = f"{type(error).__name__}: {str(error)[:300]}"
    record["t_end"] = time.perf_counter()
    return record


def request_body(args, prompt: dict) -> dict:
    body = {"model": args.model, "max_tokens": args.max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    if args.api == "chat":
        body["messages"] = [{"role": "user", "content": prompt["text"]}]
    else:
        body["prompt"] = prompt["text"] if args.text_prompts else prompt["ids"]
    body.update(json.loads(args.extra_body))
    return body


def summarize_steady(records: list[dict], args) -> dict:
    errors = [r["error"] for r in records if r["error"]]
    if errors:
        return {"clean": False, "errors": errors[:3]}
    t0 = min(r["t_send"] for r in records)
    t_last = max(r["chunks"][-1] for r in records if r["chunks"])
    counted = [len(r["chunks"]) for r in records]
    reported = [(r["usage"] or {}).get("completion_tokens") for r in records]
    prompt_tokens = [(r["usage"] or {}).get("prompt_tokens") for r in records]
    tokens = [rep if rep is not None else cnt for rep, cnt in zip(reported, counted)]
    # Decode window: from the moment the slowest stream has its first token to the first stream's last token.
    w0 = max(r["chunks"][0] for r in records)
    w1 = min(r["chunks"][-1] for r in records)
    in_window = sum(sum(1 for t in r["chunks"] if w0 < t <= w1) for r in records)
    # Token-based (track C): the same window, every chunk weighted by the tokens it carries (chunk_weights).
    weights = [chunk_weights(r, n) for r, n in zip(records, tokens)]
    in_window_tokens = sum(sum(w for t, w in zip(r["chunks"], ws) if w0 < t <= w1) for r, ws in zip(records, weights))
    nvml_util = args.nvml_sampler.mean(w0, w1) if getattr(args, "nvml_sampler", None) else None
    gaps = [(b - a) * 1e3 for r in records for a, b in zip(r["chunks"], r["chunks"][1:])]
    # Long decodes: the token gap by how many tokens the stream has already generated (its context grows with it).
    by_position = None
    if args.max_tokens >= 2048:
        edges = [0, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
        by_position = {}
        for lo, hi in zip(edges, edges[1:]):
            values = [(r["chunks"][i + 1] - r["chunks"][i]) * 1e3 for r in records for i in range(lo, min(hi, len(r["chunks"]) - 1))]
            if values:
                by_position[f"{lo}-{hi}"] = {"p50": pct(values, 0.50), "p99": pct(values, 0.99), "max": max(values), "n": len(values)}
    if args.allow_short:
        # Servers without ignore_eos or a per-request length: every stream must still decode for a while together.
        clean = min(tokens) >= args.min_tokens and w1 - w0 >= 1.0
    else:
        clean = all(t == args.max_tokens for t in tokens)
    if args.api == "completions" and not args.text_prompts and not getattr(args, "prompts_file", None):
        clean = clean and all(p == args.prompt_tokens for p in prompt_tokens)
    # One chunk per token is what makes token gaps meaningful; a server that batches tokens into chunks fails here.
    one_per_token = all(abs(c - t) <= max(2, 0.01 * t) for c, t in zip(counted, tokens))
    return {"clean": clean, "one_chunk_per_token": one_per_token, "tokens_from_usage": all(r is not None for r in reported),
            "tokens": tokens, "chunks": counted, "prompt_tokens": prompt_tokens,
            "finish": [r["finish"] for r in records],
            "e2e_tok_s": sum(tokens) / (t_last - t0),
            "decode_tok_s": in_window / (w1 - w0) if w1 > w0 else float("nan"),
            "decode_tok_s_by_tokens": in_window_tokens / (w1 - w0) if w1 > w0 else float("nan"),
            "tokens_per_chunk": sum(tokens) / max(1, sum(counted)),
            "chunk_tokens_reported": all(len(r.get("chunk_tokens") or []) == len(r["chunks"]) and all(k is not None for k in r["chunk_tokens"]) for r in records),
            "decode_window_s": w1 - w0, "nvml_util": nvml_util, "tpot_ms_by_generated": by_position,
            "ttft_s": [r["chunks"][0] - r["t_send"] for r in records],
            "tpot_ms": {"p50": pct(gaps, 0.50), "p90": pct(gaps, 0.90), "p99": pct(gaps, 0.99),
                        "max": max(gaps) if gaps else float("nan"), "mean": statistics.fmean(gaps) if gaps else float("nan")},
            "t_start_unix": time.time() - (time.perf_counter() - t0), "wall_s": t_last - t0}


async def steady(args) -> dict:
    tok = prompts.load(args.model_dir)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=args.read_timeout)
    connector = aiohttp.TCPConnector(limit=0)
    args.nvml_sampler = Nvml() if args.nvml else None
    repeats, texts = [], []
    natural = json.loads(Path(args.prompts_file).read_text()) if args.prompts_file else None
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for repeat in range(-args.warmup, args.repeats):
            # A different prompt for every stream of every repeat: no prefix cache hit can shorten a prefill.
            if natural is not None:   # fixed prompts of natural text, a different one for every stream of every repeat
                batch = [{"ids": natural[((repeat + args.warmup) * args.batch + i) % len(natural)], "text": None, "exact_text": True} for i in range(args.batch)]
            else:
                batch = [prompts.make(tok, args.prompt_tokens, args.seed + 10_000 * (repeat + args.warmup) + i)
                         for i in range(args.batch)]
            if args.text_prompts and args.api == "completions" and not all(p["exact_text"] for p in batch):
                raise SystemExit("a text prompt does not encode back to the requested token count")
            records = await asyncio.gather(*(one_stream(session, args.base, request_body(args, p)) for p in batch))
            summary = summarize_steady(records, args)
            summary["repeat"] = repeat
            if repeat >= 0 and args.save_text:
                texts.append(["".join(r["text"]) for r in records])
            if args.keep_chunks:
                summary["chunk_times"] = [[t - records[0]["t_send"] for t in r["chunks"]] for r in records]
            tag = "warmup" if repeat < 0 else "repeat"
            brief = {k: summary.get(k) for k in ("clean", "one_chunk_per_token", "e2e_tok_s", "decode_tok_s", "decode_tok_s_by_tokens", "tokens_per_chunk", "tpot_ms", "errors") if k in summary}
            if summary.get("tokens") and not summary.get("clean"):
                brief.update(tokens=summary["tokens"], chunks=summary["chunks"], prompt_tokens=summary["prompt_tokens"], finish=summary["finish"])
            print(f"MK_STEADY {tag} {repeat} B={args.batch} P={args.prompt_tokens} N={args.max_tokens} {json.dumps(brief)}", flush=True)
            if repeat >= 0:
                repeats.append(summary)
            await asyncio.sleep(args.pause)
    good = [r for r in repeats if r.get("clean")]

    def spread(key: str) -> dict | None:
        values = [r[key] for r in good]
        return {"median": statistics.median(values), "min": min(values), "max": max(values)} if values else None

    return {"workload": "steady", "batch": args.batch, "prompt_tokens": args.prompt_tokens, "max_tokens": args.max_tokens,
            "repeats": len(repeats), "clean_repeats": len(good), "timing": "end-to-end, client clock", "api": args.api,
            "one_chunk_per_token": all(r.get("one_chunk_per_token") for r in good) if good else None,
            "tokens_from_usage": all(r.get("tokens_from_usage") for r in good) if good else None,
            "tokens_per_stream": statistics.median([t for r in good for t in r["tokens"]]) if good else None,
            "prompt_tokens_seen": sorted({p for r in good for p in r["prompt_tokens"] if p is not None}) if good else None,
            "e2e_tok_s": spread("e2e_tok_s"), "decode_tok_s": spread("decode_tok_s"),
            # track C: decode_tok_s with every chunk weighted by its tokens (equal to decode_tok_s at one chunk per token)
            "decode_tok_s_by_tokens": spread("decode_tok_s_by_tokens"),
            "tokens_per_chunk": statistics.median([r["tokens_per_chunk"] for r in good]) if good else None,
            "tpot_p50_ms": statistics.median([r["tpot_ms"]["p50"] for r in good]) if good else None,
            "tpot_p99_ms": statistics.median([r["tpot_ms"]["p99"] for r in good]) if good else None,
            "ttft_median_s": statistics.median([t for r in good for t in r["ttft_s"]]) if good else None,
            "tpot_p50_ms_by_generated": {k: statistics.median([r["tpot_ms_by_generated"][k]["p50"] for r in good if k in (r.get("tpot_ms_by_generated") or {})])
                                         for k in (good[0].get("tpot_ms_by_generated") or {})} if good else None,
            "tpot_p99_ms_by_generated": {k: statistics.median([r["tpot_ms_by_generated"][k]["p99"] for r in good if k in (r.get("tpot_ms_by_generated") or {})])
                                         for k in (good[0].get("tpot_ms_by_generated") or {})} if good else None,
            "nvml_util_mean": statistics.median([r["nvml_util"]["mean"] for r in good if r.get("nvml_util")]) if any(r.get("nvml_util") for r in good) else None,
            "nvml_util_per_card": [round(statistics.median(v), 1) for v in zip(*[r["nvml_util"]["per_card"] for r in good if (r.get("nvml_util") or {}).get("per_card")])] or None,
            # Token-based rate over all clean repeats (tokens the server reports / wall time from the first send to the last
            # token): the figure to use when a server sends several tokens per chunk, where the chunk-based
            # decode_tok_s undercounts.
            "tokens_total": sum(sum(r["tokens"]) for r in good), "wall_total_s": sum(r["wall_s"] for r in good),
            "tok_s_by_tokens": (sum(sum(r["tokens"]) for r in good) / sum(r["wall_s"] for r in good)) if good else None,
            "tok_s_by_tokens_per_repeat": [sum(r["tokens"]) / r["wall_s"] for r in good],
            "prompts": "natural text (prompts file)" if natural is not None else "synthetic", "texts": texts if args.save_text else None,
            "per_repeat": repeats}


async def arrivals(args) -> dict:
    tok = prompts.load(args.model_dir)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=args.read_timeout)

    def long_body(seed: int) -> dict:
        prompt = prompts.make(tok, args.arrival_tokens, seed)
        if args.text_prompts and not prompt["exact_text"]:
            raise SystemExit("a text prompt does not encode back to the requested token count")
        return request_body(argparse.Namespace(**{**vars(args), "max_tokens": args.arrival_max_tokens}), prompt)

    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        alone = []
        for i in range(args.alone_repeats + 1):  # the first one warms the server and is dropped
            record = await one_stream(session, args.base, long_body(args.seed + 900_000 + i))
            if record["error"] or not record["chunks"]:
                return {"workload": "arrivals", "clean": False, "errors": [record["error"] or "no tokens"], "phase": "alone"}
            alone.append(record["chunks"][0] - record["t_send"])
            await asyncio.sleep(1.0)
        alone = alone[1:]
        ttft_alone = statistics.median(alone)
        print(f"MK_ARRIVALS alone ttft_s={[round(x, 3) for x in alone]}", flush=True)

        stop_at = time.perf_counter() + args.warmup_s + args.duration_s
        background: list[list[float]] = [[] for _ in range(args.streams)]
        errors: list[str] = []

        async def loop_stream(index: int) -> None:
            turn = 0
            while time.perf_counter() < stop_at:
                prompt = prompts.make(tok, args.prompt_tokens, args.seed + 1_000 * index + turn)
                record = await one_stream(session, args.base, request_body(args, prompt))
                if record["error"]:
                    errors.append(record["error"])
                    await asyncio.sleep(0.5)
                background[index].extend(record["chunks"])
                turn += 1

        tasks = [asyncio.create_task(loop_stream(i)) for i in range(args.streams)]
        t_begin = time.perf_counter()
        await asyncio.sleep(args.warmup_s)
        arrived = []
        k = 0
        while time.perf_counter() + args.period_s * 0.5 < stop_at:
            record = await one_stream(session, args.base, long_body(args.seed + 500_000 + k))
            arrived.append(record)
            ttft = (record["chunks"][0] - record["t_send"]) if record["chunks"] else float("nan")
            print(f"MK_ARRIVALS arrival {k} ttft_s={ttft:.3f} error={record['error']}", flush=True)
            k += 1
            await asyncio.sleep(max(0.0, record["t_send"] + args.period_s - time.perf_counter()))
        stop_at = time.perf_counter()  # let each background stream finish the request it is on
        await asyncio.gather(*tasks)
        t_stop = max((b[-1] for b in background if b), default=t_begin)

    good = [r for r in arrived if not r["error"] and r["chunks"]]
    windows = [(r["t_send"], r["chunks"][0]) for r in good]            # prefill: sent -> first token
    busy = [(r["t_send"], r.get("t_end", r["chunks"][-1])) for r in good]  # whole life of an arrival
    t_measure = t_begin + args.warmup_s * 0.5
    tokens = sorted(t for b in background for t in b if t_measure <= t <= t_stop)
    inside = sum(1 for t in tokens if any(a <= t <= b for a, b in windows))
    outside = sum(1 for t in tokens if not any(a <= t <= b for a, b in busy))
    window_s = sum(b - a for a, b in windows)
    quiet_s = (t_stop - t_measure) - sum(min(b, t_stop) - max(a, t_measure) for a, b in busy)
    rate_in = inside / window_s if window_s > 0 else float("nan")
    rate_out = outside / quiet_s if quiet_s > 0 else float("nan")
    gaps_all, gaps_in = [], []
    for b in background:
        for x, y in zip(b, b[1:]):
            if x >= t_measure:
                gaps_all.append((y - x) * 1e3)
                if any(a <= y and x <= c for a, c in windows):
                    gaps_in.append((y - x) * 1e3)

    def gap_stats(gaps: list[float]) -> dict:
        return {"p50": pct(gaps, 0.5), "p99": pct(gaps, 0.99), "max": max(gaps) if gaps else float("nan"), "n": len(gaps)}

    ttfts = [r["chunks"][0] - r["t_send"] for r in good]
    return {"workload": "arrivals", "clean": not errors and len(good) == len(arrived) and len(good) > 0,
            "errors": errors[:3] + [r["error"] for r in arrived if r["error"]][:3],
            "streams": args.streams, "arrival_tokens": args.arrival_tokens, "period_s": args.period_s,
            "duration_s": args.duration_s, "arrivals": len(arrived), "timing": "end-to-end, client clock",
            "ttft_alone_s": ttft_alone, "ttft_alone_all_s": alone, "ttft_loaded_s": ttfts,
            "ttft_ratio": {"median": statistics.median(ttfts) / ttft_alone, "max": max(ttfts) / ttft_alone} if ttfts else None,
            "background_tok_s_outside": rate_out, "background_tok_s_in_prefill": rate_in,
            "decode_kept": rate_in / rate_out if rate_out and rate_out == rate_out else float("nan"),
            "prefill_window_s": window_s, "gap_ms_all": gap_stats(gaps_all), "gap_ms_in_prefill": gap_stats(gaps_in)}


async def churn(args) -> dict:
    tok = prompts.load(args.model_dir)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=args.read_timeout)
    sampler = Nvml() if args.nvml else None
    records: list[dict] = []
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        t_begin = time.perf_counter()
        t_measure, t_stop = t_begin + args.warmup_s, t_begin + args.warmup_s + args.duration_s

        async def worker(index: int) -> None:
            turn = 0
            while time.perf_counter() < t_stop:
                prompt = prompts.make_with_prefix(tok, args.prefix_tokens, args.prompt_tokens, args.seed + 100_000 * index + turn)
                record = await one_stream(session, args.base, request_body(args, prompt))
                records.append(record)
                if record["error"]:
                    await asyncio.sleep(0.2)
                turn += 1

        await asyncio.gather(*(worker(i) for i in range(args.batch)))
    errors = [r["error"] for r in records if r["error"]]
    done = [r for r in records if not r["error"] and r["chunks"] and t_measure <= r["t_send"] and r["chunks"][-1] <= t_stop]
    tokens_in_window = sum(1 for r in records for t in r["chunks"] if t_measure <= t <= t_stop)
    ttft = [r["chunks"][0] - r["t_send"] for r in done]
    gaps = [(b - a) * 1e3 for r in done for a, b in zip(r["chunks"], r["chunks"][1:])]
    latency = [r["chunks"][-1] - r["t_send"] for r in done]
    usage = [r["usage"] or {} for r in done]
    cached = [(u.get("prompt_tokens_details") or {}).get("cached_tokens") for u in usage]
    cached = [c for c in cached if c is not None]
    short = sum(1 for r, u in zip(done, usage) if (u.get("completion_tokens") or len(r["chunks"])) != args.max_tokens)
    util = sampler.mean(t_measure, t_stop) if sampler else None
    return {"workload": "churn", "batch": args.batch, "prefix_tokens": args.prefix_tokens, "prompt_tokens": args.prompt_tokens,
            "max_tokens": args.max_tokens, "duration_s": args.duration_s, "timing": "end-to-end, client clock",
            "clean": not errors and len(done) > 0 and short == 0, "errors": errors[:3], "requests_done": len(done),
            "requests_short": short, "requests_per_s": len(done) / args.duration_s, "output_tok_s": tokens_in_window / args.duration_s,
            "ttft_ms": {"p50": pct(ttft, 0.5) * 1e3, "p99": pct(ttft, 0.99) * 1e3, "max": max(ttft) * 1e3} if ttft else None,
            "gap_ms": {"p50": pct(gaps, 0.5), "p99": pct(gaps, 0.99), "max": max(gaps)} if gaps else None,
            "request_latency_ms": {"p50": pct(latency, 0.5) * 1e3, "p99": pct(latency, 0.99) * 1e3} if latency else None,
            "cached_prompt_tokens_mean": statistics.fmean(cached) if cached else None,
            "prompt_tokens_mean": statistics.fmean([u.get("prompt_tokens") or 0 for u in usage]) if usage else None,
            "nvml_util_mean": util["mean"] if util else None}


async def replay(args) -> dict:
    tok = prompts.load(args.model_dir)
    pool = prompts.word_pool(tok)
    rows = [json.loads(line) for line in Path(args.trace).read_text().splitlines() if line.strip()]
    t_first = rows[0]["timestamp"]
    plan, cut = [], 0
    for row in rows:
        at = (row["timestamp"] - t_first) / 1e3 / args.rate_scale
        if at > args.duration_s:
            break
        n_in = min(row["input_length"], args.max_input_tokens)
        cut += n_in < row["input_length"]
        n_out = max(1, min(row["output_length"], args.max_output_tokens))
        plan.append((at, prompts.from_block_hashes(pool, row["hash_ids"], n_in), n_out, row["input_length"]))
    timeout = aiohttp.ClientTimeout(total=None, sock_read=args.read_timeout)
    sampler = Nvml() if args.nvml else None
    records: list[dict] = []
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        t0 = time.perf_counter()

        async def fire(at: float, ids: list[int], n_out: int) -> None:
            await asyncio.sleep(max(0.0, t0 + at - time.perf_counter()))
            body = request_body(argparse.Namespace(**{**vars(args), "max_tokens": n_out}), {"ids": ids, "text": None})
            record = await one_stream(session, args.base, body)
            record.update(planned_at=at, n_in=len(ids), n_out=n_out, lateness=record["t_send"] - (t0 + at))
            records.append(record)

        tasks = [asyncio.create_task(fire(at, ids, n_out)) for at, ids, n_out, _ in plan]
        # Stop waiting `drain_s` after the last arrival: what is unfinished then is the queue the server built up.
        done, pending = await asyncio.wait(tasks, timeout=args.duration_s + args.drain_s)
        for task in pending:
            task.cancel()
        t_end = time.perf_counter()
    good = [r for r in records if not r["error"] and r["chunks"]]
    errors = [r["error"] for r in records if r["error"]]
    ttft = [r["chunks"][0] - r["t_send"] for r in good]
    gaps = [(b - a) * 1e3 for r in good for a, b in zip(r["chunks"], r["chunks"][1:])]
    # Per-request mean time between tokens after the first: the trace paper's TBT.
    tbt = [(r["chunks"][-1] - r["chunks"][0]) / max(1, len(r["chunks"]) - 1) for r in good if len(r["chunks"]) > 1]
    usage = [r["usage"] or {} for r in good]
    cached = sum((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0 for u in usage)
    prompt_total = sum(u.get("prompt_tokens") or 0 for u in usage)
    reports_cache = any((u.get("prompt_tokens_details") or {}).get("cached_tokens") is not None for u in usage)
    within = sum(1 for r in good if r["chunks"][0] - r["t_send"] <= args.slo_ttft_s
                 and (len(r["chunks"]) < 2 or (r["chunks"][-1] - r["chunks"][0]) / (len(r["chunks"]) - 1) <= args.slo_tbt_s))
    out_tokens = sum(len(r["chunks"]) for r in good)
    util = sampler.mean(t0, t0 + args.duration_s) if sampler else None

    def pcts(values: list[float], scale: float = 1.0) -> dict | None:
        return {"p50": pct(values, 0.5) * scale, "p90": pct(values, 0.9) * scale, "p99": pct(values, 0.99) * scale,
                "max": max(values) * scale} if values else None

    return {"workload": "replay", "trace": Path(args.trace).name, "rate_scale": args.rate_scale, "duration_s": args.duration_s,
            "timing": "end-to-end, client clock; open loop at the trace's arrival times / rate_scale",
            "planned": len(plan), "offered_req_s": len(plan) / args.duration_s, "inputs_cut": cut,
            "input_tokens_mean": statistics.fmean([len(p[1]) for p in plan]) if plan else None,
            "output_tokens_mean": statistics.fmean([p[2] for p in plan]) if plan else None,
            "finished": len(good), "errors": len(errors), "error_examples": errors[:2], "unfinished_at_stop": len(pending),
            "clean": not errors and len(plan) > 0, "saturated": len(pending) > 0.05 * max(1, len(plan)),
            "completed_req_s": len(good) / (t_end - t0), "output_tok_s": out_tokens / (t_end - t0),
            "ttft_ms": pcts(ttft, 1e3), "gap_ms": pcts(gaps), "tbt_ms": pcts(tbt, 1e3),
            "slo": {"ttft_s": args.slo_ttft_s, "tbt_s": args.slo_tbt_s, "within": within, "share_of_planned": within / max(1, len(plan))},
            "cached_prompt_share": (cached / prompt_total) if reports_cache and prompt_total else None,
            "client_lateness_ms_max": max((r["lateness"] for r in records), default=0.0) * 1e3,
            "nvml_util_mean": util["mean"] if util else None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", choices=["steady", "arrivals", "churn", "replay"])
    parser.add_argument("--trace", default=None, help="replay: JSON-lines trace (timestamp, input_length, output_length, hash_ids)")
    parser.add_argument("--rate-scale", type=float, default=1.0, help="replay: compress the trace's time by this factor")
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--drain-s", type=float, default=60.0, help="replay: how long to wait for stragglers after the last arrival")
    parser.add_argument("--slo-ttft-s", type=float, default=5.0)
    parser.add_argument("--slo-tbt-s", type=float, default=0.1)
    parser.add_argument("--prefix-tokens", type=int, default=0, help="churn: tokens shared by every request (prefix-cacheable)")
    parser.add_argument("--api", choices=["completions", "chat"], default="completions",
                        help="chat: /v1/chat/completions with the prompt as one user message (the template adds tokens)")
    parser.add_argument("--allow-short", action="store_true",
                        help="accept streams that stop before max_tokens (servers without ignore_eos)")
    parser.add_argument("--min-tokens", type=int, default=128, help="with --allow-short: fewest tokens a stream may produce")
    parser.add_argument("--nvml", action="store_true", help="steady: sample NVML GPU utilization over each decode window")
    parser.add_argument("--streams", type=int, default=7, help="arrivals: background streams")
    parser.add_argument("--arrival-tokens", type=int, default=8192, help="arrivals: length of the long prompt")
    parser.add_argument("--arrival-max-tokens", type=int, default=16)
    parser.add_argument("--period-s", type=float, default=20.0)
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--warmup-s", type=float, default=20.0)
    parser.add_argument("--alone-repeats", type=int, default=3)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True, help="model name the server expects")
    parser.add_argument("--model-dir", required=True, help="directory with tokenizer.json")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between repeats")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--read-timeout", type=float, default=600.0)
    parser.add_argument("--text-prompts", action="store_true", help="send text, for servers that take no token ids")
    parser.add_argument("--extra-body", default="{}", help="JSON merged into every request (for example ignore_eos)")
    parser.add_argument("--prompts-file", default=None, help="steady: JSON list of token-id lists used as prompts instead of the synthetic ones")
    parser.add_argument("--save-text", action="store_true", help="steady: keep every stream's text in the result (to compare systems token for token)")
    parser.add_argument("--keep-chunks", action="store_true", help="store every token's arrival time")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.workload == "replay":
        result = asyncio.run(replay(args))
        result.update(label=args.label, extra_body=json.loads(args.extra_body))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
        print("MK_REPLAY " + json.dumps(result), flush=True)
        return 0 if result.get("clean") else 3
    if args.workload == "churn":
        result = asyncio.run(churn(args))
        result.update(label=args.label, extra_body=json.loads(args.extra_body))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
        print("MK_CHURN " + json.dumps(result), flush=True)
        return 0 if result.get("clean") else 3
    if args.workload == "arrivals":
        result = asyncio.run(arrivals(args))
        result.update(label=args.label, extra_body=json.loads(args.extra_body))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
        print("MK_RESULT " + json.dumps(result), flush=True)
        return 0 if result.get("clean") else 3
    result = asyncio.run(steady(args))
    result["label"] = args.label
    result["extra_body"] = json.loads(args.extra_body)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    brief = {k: v for k, v in result.items() if k != "per_repeat"}
    print("MK_RESULT " + json.dumps(brief), flush=True)
    return 0 if result["clean_repeats"] == result["repeats"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
