"""python check_clients.py URL: mkserve's APIs through the official SDKs (openai, anthropic, ollama), against the fake
engine's scripted answers (make_script.py). Prints one CHECK line per check; exits 1 if any fails."""
import json
import sys

import anthropic
import ollama
import openai

URL = sys.argv[1].rstrip("/")
failed = []


def check(name, cond, detail=""):
    print(f"CHECK {'ok  ' if cond else 'FAIL'} {name}" + ("" if cond else f": {detail}"), flush=True)
    if not cond:
        failed.append(name)


WEATHER_TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
                                                 "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
                                                                "required": ["city"]}}}
HELLO = "Hello! 你好，有什么可以帮你？"
THINK = "The user greets me. 你好 needs a reply."

# ---------------------------------------------------------------------------------------------- OpenAI chat
oa = openai.OpenAI(base_url=URL + "/v1", api_key="none")
model = oa.models.list().data[0].id
check("openai models", bool(model))
r = oa.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=200)
m = r.choices[0].message
check("openai chat text", m.content == HELLO, repr(m.content))
check("openai chat reasoning_content", (m.model_extra or {}).get("reasoning_content") == THINK, repr(m.model_extra))
check("openai chat finish stop", r.choices[0].finish_reason == "stop", r.choices[0].finish_reason)
check("openai chat usage", r.usage.completion_tokens == 32 - 1, r.usage)  # the end token is not counted

text, think, chunks = "", "", 0
for ch in oa.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=200, stream=True,
                                     stream_options={"include_usage": True}):
    chunks += 1
    if ch.choices:
        d = ch.choices[0].delta
        text += d.content or ""
        think += (d.model_extra or {}).get("reasoning_content") or ""
check("openai chat stream text", text == HELLO, repr(text))
check("openai chat stream reasoning", think == THINK, repr(think))

r = oa.chat.completions.create(model=model, messages=[{"role": "user", "content": "What is the weather in Paris?"}], tools=[WEATHER_TOOL], max_tokens=200)
tc = r.choices[0].message.tool_calls
check("openai tool call", tc and tc[0].function.name == "get_weather" and json.loads(tc[0].function.arguments) == {"city": "Paris", "unit": "C"},
      r.choices[0].message)
check("openai tool finish", r.choices[0].finish_reason == "tool_calls", r.choices[0].finish_reason)

calls = {}
for ch in oa.chat.completions.create(model=model, messages=[{"role": "user", "content": "What is the weather in Paris?"}], tools=[WEATHER_TOOL],
                                     max_tokens=200, stream=True):
    for t in (ch.choices[0].delta.tool_calls or []) if ch.choices else []:
        c = calls.setdefault(t.index, {"name": "", "args": ""})
        c["name"] += t.function.name or ""
        c["args"] += t.function.arguments or ""
check("openai tool call stream", calls.get(0, {}).get("name") == "get_weather" and json.loads(calls[0]["args"])["city"] == "Paris", calls)

msgs = [{"role": "user", "content": "What is the weather in Paris?"}, r.choices[0].message.model_dump(exclude_none=True),
        {"role": "tool", "tool_call_id": tc[0].id, "content": "22C and sunny"}]
r2 = oa.chat.completions.create(model=model, messages=msgs, tools=[WEATHER_TOOL], max_tokens=200)
check("openai tool round trip", r2.choices[0].message.content == "It is 22C and sunny in Paris.", r2.choices[0].message)

r = oa.chat.completions.create(model=model, messages=[{"role": "user", "content": "count to five"}], stop=["STOP"], max_tokens=200)
check("openai stop string", r.choices[0].message.content == "one two three " and r.choices[0].finish_reason == "stop", r.choices[0])
r = oa.chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=5)
check("openai max_tokens", r.choices[0].finish_reason == "length" and r.usage.completion_tokens == 5, r)

r = oa.completions.create(model=model, prompt="Once upon a time", max_tokens=200)
check("openai completions (raw text, special tokens left out)", r.choices[0].text.startswith("The user greets me."), repr(r.choices[0].text))

# ------------------------------------------------------------------------------------------ OpenAI Responses
r = oa.responses.create(model=model, input="hi", max_output_tokens=200)
check("responses text", r.output_text == HELLO, r.output_text)
check("responses reasoning item", any(i.type == "reasoning" for i in r.output), [i.type for i in r.output])
r = oa.responses.create(model=model, input="What is the weather in Paris?", tools=[{"type": "function", **WEATHER_TOOL["function"]}],
                        max_output_tokens=200)
fc = [i for i in r.output if i.type == "function_call"]
check("responses function_call", fc and fc[0].name == "get_weather" and json.loads(fc[0].arguments)["city"] == "Paris", r.output)
r2 = oa.responses.create(model=model, previous_response_id=r.id, tools=[{"type": "function", **WEATHER_TOOL["function"]}],
                         input=[{"type": "function_call_output", "call_id": fc[0].call_id, "output": "22C and sunny"}], max_output_tokens=200)
check("responses previous_response_id + function_call_output", r2.output_text == "It is 22C and sunny in Paris.", r2.output_text)
types, text = [], ""
with oa.responses.stream(model=model, input="hi", max_output_tokens=200) as s:
    for ev in s:
        types.append(ev.type)
        if ev.type == "response.output_text.delta":
            text += ev.delta
    final = s.get_final_response()
check("responses stream", text == HELLO and final.output_text == HELLO and types[-1] == "response.completed", (text, types[-3:]))

# ------------------------------------------------------------------------------------------------ Anthropic
an = anthropic.Anthropic(base_url=URL, api_key="none")
r = an.messages.create(model="inkling-2-mini", max_tokens=200, messages=[{"role": "user", "content": "hi"}])
kinds = [b.type for b in r.content]
check("anthropic blocks", kinds == ["thinking", "text"], kinds)
check("anthropic text", r.content[-1].text == HELLO and r.stop_reason == "end_turn", r)
TOOLS_A = [{"name": "get_weather", "description": "Current weather of a city", "input_schema": WEATHER_TOOL["function"]["parameters"]}]
with an.messages.stream(model="inkling-2-mini", max_tokens=200, tools=TOOLS_A,
                        messages=[{"role": "user", "content": "What is the weather in Paris?"}]) as s:
    msg = s.get_final_message()
tu = [b for b in msg.content if b.type == "tool_use"]
check("anthropic stream tool_use", tu and tu[0].name == "get_weather" and tu[0].input == {"city": "Paris", "unit": "C"} and msg.stop_reason == "tool_use",
      msg)
r = an.messages.create(model="inkling-2-mini", max_tokens=200, tools=TOOLS_A, system="You are terse.", messages=[
    {"role": "user", "content": "What is the weather in Paris?"},
    {"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in msg.content]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu[0].id, "content": "22C and sunny"}]}])
check("anthropic tool round trip", r.content[-1].text == "It is 22C and sunny in Paris.", r.content)
n = an.messages.count_tokens(model="inkling-2-mini", messages=[{"role": "user", "content": "hi"}])
check("anthropic count_tokens", n.input_tokens > 0, n)
r = an.messages.create(model="inkling-2-mini", max_tokens=200, stop_sequences=["STOP"], messages=[{"role": "user", "content": "count to five"}])
check("anthropic stop_sequence", r.stop_reason == "stop_sequence" and r.stop_sequence == "STOP" and r.content[-1].text == "one two three ", r)

# --------------------------------------------------------------------------------------------------- Ollama
ol = ollama.Client(host=URL)
check("ollama list", ol.list().models[0].model == model, ol.list())
text, think = "", ""
for part in ol.chat(model=model, messages=[{"role": "user", "content": "hi"}], stream=True, think=True):
    text += part.message.content or ""
    think += part.message.thinking or ""
check("ollama chat stream", text == HELLO and think == THINK, (text, think))
r = ol.chat(model=model, messages=[{"role": "user", "content": "What is the weather in Paris?"}], tools=[WEATHER_TOOL])
check("ollama tool call", r.message.tool_calls and r.message.tool_calls[0].function.name == "get_weather"
      and r.message.tool_calls[0].function.arguments == {"city": "Paris", "unit": "C"}, r.message)
r2 = ol.chat(model=model, tools=[WEATHER_TOOL], messages=[{"role": "user", "content": "What is the weather in Paris?"}, r.message,
                                                          {"role": "tool", "tool_name": "get_weather", "content": "22C and sunny"}])
check("ollama tool round trip", r2.message.content == "It is 22C and sunny in Paris.", r2.message)
r = ol.generate(model=model, prompt="hi")
check("ollama generate", r.response == HELLO and r.done, r)

print("FAILED: " + ", ".join(failed) if failed else "ALL CLIENT CHECKS PASSED")
sys.exit(1 if failed else 0)
