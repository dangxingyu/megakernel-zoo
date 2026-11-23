"""Sampling over mkserve's chat API: tokens per second by temperature (after the first streamed token), the same seed twice."""
import json, sys, time, urllib.request
port = int(sys.argv[1])
def chat(q, **kw):
    body = {"model": "m", "messages": [{"role": "user", "content": q}], "max_tokens": 400, "stream": True, "stream_options": {"include_usage": True},
            "reasoning_effort": "none", **kw}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; text = ""; usage = {}
    with urllib.request.urlopen(req) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                dl = c.get("delta", {}); piece = (dl.get("content") or "") + (dl.get("reasoning_content") or "")
                if piece and first is None: first = time.perf_counter()
                text += piece
    n = usage.get("completion_tokens", 0)
    return text, n, (n - 1) / (time.perf_counter() - first) if first and n > 1 else 0.0
for q in ("Write a short story about a lighthouse keeper who finds a message in a bottle.", "Explain how a hash table works, with a small Python example."):
    for name, kw in (("temperature 0", {"temperature": 0}), ("temperature 0.7", {"temperature": 0.7, "seed": 5}), ("temperature 1.0, top_p 0.95", {"temperature": 1.0, "top_p": 0.95, "seed": 5})):
        chat(q, **kw)
        text, n, tps = chat(q, **kw)
        again = chat(q, **kw)[0] == text
        print(json.dumps({"question": q[:28], "sampling": name, "tokens": n, "tokens_per_s": round(tps, 1), "same_text_again": again, "text": text[:60]}, ensure_ascii=False), flush=True)
