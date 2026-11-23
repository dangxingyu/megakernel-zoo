"""python check_prompts.py MODEL_DIR PROMPTS.jsonl: the prompts mkserve rendered (logged by the fake engine) against
transformers' apply_chat_template for the same conversations: byte for byte."""
import json
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
seen = [json.loads(l)["prompt"] for l in open(sys.argv[2])]
TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
                                         "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
                                                        "required": ["city"]}}}
call = {"id": "x", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "C"}}}
cases = {
    "plain chat": dict(conversation=[{"role": "user", "content": "hi"}]),
    "tools": dict(conversation=[{"role": "user", "content": "What is the weather in Paris?"}], tools=[TOOL]),
    "tool round trip (OpenAI shape)": dict(conversation=[
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "content": "", "reasoning_content": "I should call the weather tool.", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "x", "content": "22C and sunny"}], tools=[TOOL]),
    "system + tool round trip (Anthropic shape)": dict(conversation=[
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "content": "", "reasoning_content": "I should call the weather tool.", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "x", "name": "get_weather", "content": "22C and sunny"}], tools=[TOOL]),
}
bad = 0
for name, kw in cases.items():
    want = tok.apply_chat_template(tokenize=False, **kw)
    ok = want in seen
    bad += not ok
    print(f"CHECK {'ok  ' if ok else 'FAIL'} prompt: {name}" + ("" if ok else f"\n  want {want!r}\n  closest {min(seen, key=lambda s: abs(len(s) - len(want)))!r}"))
print("ALL PROMPT CHECKS PASSED" if not bad else f"{bad} PROMPT CHECKS FAILED")
sys.exit(1 if bad else 0)
