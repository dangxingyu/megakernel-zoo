"""A chat with a long system prompt over mkserve's OpenAI API, turn after turn: time to the first chunk and the prompt
tokens the engine kept from its prefix cache (usage.prompt_tokens_details.cached_tokens)."""
import os as _os
# paths: the repository this file is in (or MK_ZOO), the model snapshot (MK_MODEL_DIR), the home directory
ZOO = (_os.environ.get("MK_ZOO") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))) + "/"
MODEL_DIR = _os.environ.get("MK_MODEL_DIR") or _os.path.expanduser("~/models/Inkling-2-Mini")
HOME = _os.path.expanduser("~")
import json, sys, time, urllib.request
port, effort = int(sys.argv[1]), (sys.argv[2] if len(sys.argv) > 2 else "none")
R = ZOO
system = "You are a helpful assistant for this project. Its notes follow.\n\n" + open(R + "runs-mac/deviation/README.md").read()[:5500]
questions = ["In one sentence: what causes most of the error?", "And which family contributes least?", "Give that as a 5-word headline.",
             "Now in Chinese."]
messages = [{"role": "system", "content": system}]
for qn in questions:
    messages.append({"role": "user", "content": qn})
    body = {"model": "thinkingmachines/Inkling-2-Mini", "messages": messages, "max_tokens": 200, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "reasoning_effort": effort}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; content = ""; reasoning = ""; usage = {}
    with urllib.request.urlopen(req) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                if first is None and (dl.get("content") or dl.get("reasoning_content")): first = time.perf_counter() - t0
                content += dl.get("content") or ""; reasoning += dl.get("reasoning_content") or ""
    total = time.perf_counter() - t0
    print(json.dumps({"turn": len(messages) // 2, "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                      "ttft_ms": round((first or -1) * 1e3), "seconds": round(total, 2), "completion_tokens": usage.get("completion_tokens"), "reply": content[:70]}, ensure_ascii=False), flush=True)
    messages.append({"role": "assistant", "content": content})
