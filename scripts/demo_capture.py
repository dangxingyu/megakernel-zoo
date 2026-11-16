#!/usr/bin/env python
"""python scripts/demo_capture.py --port P --served NAME --system LABEL --out FILE: ask a running OpenAI-compatible server
three questions over the streaming chat endpoint and record when each piece of the answer arrives.

For the demo: the recording is replayed as it happened (dashboard/demo.py). One warm-up question is not recorded. Each
record: the question, [[seconds since the request, text], ...], prompt and completion token counts as the server reports
them. Greedy decoding, the model's own chat template, answers end where the model ends them (at most --max-tokens)."""
import argparse
import http.client
import json
import time

QUESTIONS = [
    "Explain how a hash table works, and say when its operations become slow.",
    "Write a Python function that merges two sorted lists into one sorted list, then explain its running time.",
    "Why is the sky blue during the day and red at sunset? Explain it to a curious twelve-year-old.",
]
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--served", required=True)
ap.add_argument("--system", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=400)
args = ap.parse_args()


def ask(question: str) -> dict:
    conn = http.client.HTTPConnection("127.0.0.1", args.port, timeout=600)
    body = {"model": args.served, "messages": [{"role": "user", "content": question}], "max_tokens": args.max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        return {"question": question, "error": f"HTTP {resp.status}: {resp.read()[:300].decode(errors='replace')}"}
    events, usage = [], {}
    for raw in resp:
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        obj = json.loads(line[6:])
        usage = obj.get("usage") or usage
        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content") or delta.get("reasoning_content")   # a server may send the model's reasoning in a field of its own
            if text:
                events.append([round(time.perf_counter() - t0, 4), text])
    conn.close()
    return {"question": question, "events": events, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "seconds": round(time.perf_counter() - t0, 3)}


ask("Say hello in one short sentence.")  # warm-up
records = [ask(q) for q in QUESTIONS]
json.dump({"system": args.system, "records": records}, open(args.out, "w"))
for r in records:
    ev = r.get("events") or []
    rate = (len(ev) - 1) / (ev[-1][0] - ev[0][0]) if len(ev) > 1 and ev[-1][0] > ev[0][0] else 0
    print("MK_DEMO " + json.dumps({"system": args.system, "question": r["question"][:40], "error": r.get("error"), "chunks": len(ev), "completion_tokens": r.get("completion_tokens"),
                                   "first_s": ev[0][0] if ev else None, "seconds": r.get("seconds"), "chunks_per_s": round(rate, 1), "text": "".join(t for _, t in ev)[:120]}), flush=True)
