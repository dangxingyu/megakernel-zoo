#!/usr/bin/env python3
"""python3 scripts/mkserve_check.py --port P --engine NAME: use a running `mkserve serve` the way a person would
(completions and chat, streaming and not, stop strings, end of sequence, several requests at once, sampling) and
print one MK_SERVE_TEST line per check. Standard library only."""
import argparse
import http.client
import json
import threading
import time

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--engine", default="")
parser.add_argument("--model", default="Qwen/Qwen3-8B")
args = parser.parse_args()


def call(path, body=None, stream=False):
    """Returns (object, seconds) or, for a stream, (chunks, seconds to the first content, seconds in total)."""
    conn = http.client.HTTPConnection("127.0.0.1", args.port, timeout=600)
    t0 = time.perf_counter()
    if body is None:
        conn.request("GET", path)
    else:
        conn.request("POST", path, json.dumps(dict(body, model=args.model, stream=stream)), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if not stream:
        data = resp.read()
        return {"status": resp.status, **json.loads(data)}, time.perf_counter() - t0
    chunks, first = [], None
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        obj = json.loads(line[6:])
        if first is None and obj.get("choices") and (obj["choices"][0].get("text") or (obj["choices"][0].get("delta") or {}).get("content")
                                                          or (obj["choices"][0].get("delta") or {}).get("reasoning_content")):
            first = time.perf_counter() - t0
        chunks.append(obj)
    return chunks, first, time.perf_counter() - t0


def report(name, ok, **facts):
    print("MK_SERVE_TEST " + json.dumps({"engine": args.engine, "check": name, "pass": bool(ok), **facts}, ensure_ascii=False), flush=True)


def text_of(chunks, chat, field="content"):
    return "".join(((c["choices"][0].get("delta") or {}).get(field) or "") if chat else c["choices"][0].get("text", "") for c in chunks if c.get("choices"))


def answered(message):
    """A chat reply that says something: its text, or for a model that always reasons first (Inkling 2 reasons even at
    reasoning_effort "none") and ran out of max_tokens while reasoning, its reasoning."""
    return message.get("content") or message.get("reasoning_content")


def guard(name, fn):
    try:
        fn()
    except Exception as exc:
        report(name, False, error=repr(exc)[:300])


health, _ = call("/health")
mode = health.get("mode")
models, _ = call("/v1/models")
report("models", models["status"] == 200 and models["data"][0]["id"] == args.model, mode=mode, served=models["data"][0])
samples = models["data"][0].get("sampling", mode == "step")


def completion_plain():
    r, s = call("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0})
    c = r["choices"][0]
    report("completion", r["status"] == 200 and c["finish_reason"] == "length" and r["usage"]["completion_tokens"] == 8 and len(c["text"]) > 0,
           text=c["text"], usage=r["usage"], seconds=round(s, 3))


def completion_stream():
    body = {"prompt": "The capital of France is", "max_tokens": 32, "temperature": 0}
    r, _ = call("/v1/completions", body)
    chunks, first, total = call("/v1/completions", dict(body, stream_options={"include_usage": True}), stream=True)
    joined = text_of(chunks, False)
    report("completion_stream", joined == r["choices"][0]["text"] and chunks[-1].get("usage", {}).get("completion_tokens") == 32,
           same_text_as_non_stream=joined == r["choices"][0]["text"], chunks=len(chunks), ttft_s=round(first or -1, 3), seconds=round(total, 3), text=joined[:80])


CHAT = {"messages": [{"role": "user", "content": "What is 17 times 23? Reply with the number only."}], "max_tokens": 64, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False}}


def chat_plain():
    r, s = call("/v1/chat/completions", CHAT)
    c = r["choices"][0]
    # a model that reasons aloud first may use all 64 tokens: the answer has to be in the text, the turn need not have ended
    report("chat", r["status"] == 200 and "391" in c["message"]["content"] and r["usage"]["completion_tokens"] <= 64,
           content=c["message"]["content"], finish=c["finish_reason"], usage=r["usage"], seconds=round(s, 3))


def chat_stream():
    r, _ = call("/v1/chat/completions", CHAT)
    chunks, first, total = call("/v1/chat/completions", CHAT, stream=True)
    joined = text_of(chunks, True)
    report("chat_stream", joined == r["choices"][0]["message"]["content"] and chunks[0]["choices"][0]["delta"].get("role") == "assistant"
           and chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length"), content=joined, chunks=len(chunks), ttft_s=round(first or -1, 3), seconds=round(total, 3))


def stop_string():
    # whatever this model continues the prompt with: a piece from the middle of that continuation, used as a stop string, must cut it there
    body = {"prompt": "1, 2, 3, 4, 5, 6,", "max_tokens": 40, "temperature": 0}
    free, _ = call("/v1/completions", body)
    text = free["choices"][0]["text"]
    stop = text[len(text) // 2:len(text) // 2 + 3]
    r, _ = call("/v1/completions", dict(body, stop=[stop]))
    c = r["choices"][0]
    report("stop_string", bool(stop) and c["finish_reason"] == "stop" and c["text"] == text[:text.index(stop)], stop=stop, text=c["text"], free=text, usage=r["usage"])


def concurrent():
    questions = ["Name one prime number.", "Write one sentence about the sea and one about the mountains.",
                 "List three colours, separated by commas, and say which one is your favourite and why in a short sentence.", "Say hello."]
    out = [None] * len(questions)

    def one(i):
        out[i] = call("/v1/chat/completions", {"messages": [{"role": "user", "content": questions[i]}], "max_tokens": 48, "temperature": 0,
                                                "chat_template_kwargs": {"enable_thinking": False}})
    t0 = time.perf_counter()
    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(questions))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    wall = time.perf_counter() - t0
    report("concurrent_mixed_lengths", all(o and o[0]["status"] == 200 and answered(o[0]["choices"][0]["message"]) for o in out),
           wall_s=round(wall, 3), each_s=[round(o[1], 3) for o in out], prompt_tokens=[o[0]["usage"]["prompt_tokens"] for o in out],
           finishes=[o[0]["choices"][0]["finish_reason"] for o in out], first=answered(out[0][0]["choices"][0]["message"])[:60],
           reasoning_only=[not o[0]["choices"][0]["message"].get("content") for o in out])


def sampling():
    body = {"prompt": "Once upon a time", "max_tokens": 24, "temperature": 0.9, "top_p": 0.95}
    a, _ = call("/v1/completions", dict(body, seed=1))
    b, _ = call("/v1/completions", dict(body, seed=2))
    a2, _ = call("/v1/completions", dict(body, seed=1))
    g, _ = call("/v1/completions", dict(body, temperature=0))
    ta, tb, ta2, tg = (x["choices"][0]["text"] for x in (a, b, a2, g))
    # a server that samples (step mode, or a native engine that draws tokens itself): reproducible per seed, different
    # across seeds; a native engine without sampling decodes greedily and says so
    ok = (ta == ta2 and ta != tb) if samples else (ta == tb == tg)
    report("sampling", ok, mode=mode, seed1=ta[:60], seed2=tb[:60], seed1_again_same=ta == ta2, greedy=tg[:60])


def speed():
    ids = list(range(1000, 1064))
    for n in (512,):
        chunks, first, total = call("/v1/completions", {"prompt": ids, "max_tokens": n, "temperature": 0, "ignore_eos": True,
                                                         "stream_options": {"include_usage": True}}, stream=True)
        tokens = chunks[-1]["usage"]["completion_tokens"]
        report("speed_b1", tokens == n, prompt_tokens=64, new_tokens=tokens, ttft_s=round(first, 4), seconds=round(total, 3),
               decode_tok_s=round((tokens - 1) / (total - first), 1), ms_per_token=round(1e3 * (total - first) / (tokens - 1), 3))


def long_prompt():
    text = "Here is a note to summarise. " + "The quick brown fox jumps over the lazy dog near the river bank. " * 100 + "Summarise the note in five words."
    chunks, first, total = call("/v1/chat/completions", {"messages": [{"role": "user", "content": text}], "max_tokens": 24, "temperature": 0,
                                                          "chat_template_kwargs": {"enable_thinking": False}, "stream_options": {"include_usage": True}}, stream=True)
    usage = chunks[-1].get("usage", {})
    report("long_prompt", bool(text_of(chunks, True) or text_of(chunks, True, "reasoning_content")), prompt_tokens=usage.get("prompt_tokens"), ttft_s=round(first or -1, 3), seconds=round(total, 3), content=(text_of(chunks, True) or text_of(chunks, True, "reasoning_content"))[:80],
           reasoning_only=not text_of(chunks, True))


def over_context():
    """A request longer than the engine's context must be refused with an error, not answered with garbage."""
    ids = [1000 + (i % 5000) for i in range(max(64, int(health.get("max_context") or 2048) - 8))]
    out, _ = call("/v1/completions", {"prompt": ids, "max_tokens": 64, "temperature": 0})
    report("over_context_refused", out["status"] >= 400, status=out["status"], prompt_tokens=len(ids), message=str(out.get("error") or out.get("message") or "")[:160])


def bad_inputs():
    """Inputs an engine would not survive must be refused by the wrapper: a token id outside the vocabulary, no tokens to generate, an empty prompt."""
    a, _ = call("/v1/completions", {"prompt": [1000, 10 ** 9, 1001], "max_tokens": 8, "temperature": 0})
    b, _ = call("/v1/completions", {"prompt": [1000, -1], "max_tokens": 8, "temperature": 0})
    c, _ = call("/v1/completions", {"prompt": [1000, 1001], "max_tokens": -3, "temperature": 0})
    d, _ = call("/v1/completions", {"prompt": [], "max_tokens": 8, "temperature": 0})
    ok, _ = call("/v1/completions", {"prompt": [1000, 1001, 1002], "max_tokens": 4, "temperature": 0})   # and the server still answers afterwards
    report("bad_inputs_refused", all(r["status"] >= 400 for r in (a, b, c, d)) and ok["status"] == 200 and ok["usage"]["completion_tokens"] == 4,
           statuses=[r["status"] for r in (a, b, c, d, ok)])


for name, fn in (("over_context_refused", over_context), ("bad_inputs_refused", bad_inputs), ("completion", completion_plain), ("completion_stream", completion_stream), ("chat", chat_plain), ("chat_stream", chat_stream),
                 ("stop_string", stop_string), ("concurrent_mixed_lengths", concurrent), ("sampling", sampling), ("speed_b1", speed), ("long_prompt", long_prompt)):
    guard(name, fn)
