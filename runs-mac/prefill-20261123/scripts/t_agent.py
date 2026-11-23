"""A coding agent's kind of request over mkserve's Anthropic API: a long system prompt (project files) and tool
definitions, then a second turn. Time to the first streamed delta and the input tokens read from the prefix cache."""
import os as _os
# paths: the repository this file is in (or MK_ZOO), the model snapshot (MK_MODEL_DIR), the home directory
ZOO = (_os.environ.get("MK_ZOO") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))) + "/"
MODEL_DIR = _os.environ.get("MK_MODEL_DIR") or _os.path.expanduser("~/models/Inkling-2-Mini")
HOME = _os.path.expanduser("~")
import glob, json, sys, time, urllib.request
port, sys_chars = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 60000
R = ZOO
files = [R + "README.md", R + "mkserve/README.md", R + "eval/README.md", R + "CONTRIBUTING.md"] + sorted(glob.glob(R + "mkserve/*.py"))
system = "You are a coding agent working in this repository. Answer briefly. Its files follow.\n\n" + "\n\n".join(f"=== {f[len(R):]} ===\n" + open(f).read() for f in files)[:sys_chars]
tools = [{"name": n, "description": d, "input_schema": {"type": "object", "properties": p, "required": list(p)}} for n, d, p in (
    ("read_file", "Read a file of the repository.", {"path": {"type": "string"}}),
    ("grep", "Search the repository for a regular expression.", {"pattern": {"type": "string"}}),
    ("run", "Run a shell command and return its output.", {"command": {"type": "string"}}),
    ("edit_file", "Replace a string in a file.", {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}))]
messages = []
for qn in ("Which command checks that this machine is ready? One line.", "And which one times a running server? One line.", "Use the grep tool to find where `--max-context` is defined."):
    messages.append({"role": "user", "content": qn})
    body = {"model": "thinkingmachines/Inkling-2-Mini", "system": system, "messages": messages, "tools": tools, "max_tokens": 300, "stream": True,
            "thinking": {"type": "disabled"}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/messages", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-api-key": "mkserve", "anthropic-version": "2023-06-01"})
    t0 = time.perf_counter(); first = None; text = ""; usage = {}; calls = []; stop = None
    try:
        with urllib.request.urlopen(req) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data: "): continue
                d = json.loads(line[6:])
                if d["type"] == "message_start": usage.update(d["message"]["usage"])
                elif d["type"] == "content_block_start" and d["content_block"]["type"] == "tool_use": calls.append([d["content_block"]["name"], ""])
                elif d["type"] == "content_block_delta":
                    if first is None: first = time.perf_counter() - t0
                    dl = d["delta"]
                    text += dl.get("text", "")
                    if dl.get("type") == "input_json_delta" and calls: calls[-1][1] += dl.get("partial_json", "")
                elif d["type"] == "message_delta": usage.update(d.get("usage") or {}); stop = d["delta"].get("stop_reason")
    except urllib.error.HTTPError as e:
        print(json.dumps({"turn": len(messages) // 2 + 1, "http": e.code, "error": e.read().decode()[:300]})); break
    print(json.dumps({"turn": len(messages) // 2 + 1, "input_tokens": usage.get("input_tokens"), "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
                      "ttft_s": round(first or -1, 2), "seconds": round(time.perf_counter() - t0, 2), "output_tokens": usage.get("output_tokens"),
                      "stop_reason": stop, "text": text[:90], "tool_calls": calls}, ensure_ascii=False), flush=True)
    messages.append({"role": "assistant", "content": text or "(tool call)"})
