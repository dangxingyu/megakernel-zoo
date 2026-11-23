"""python make_script.py MODEL_DIR OUT.json: the fake engine's answers, in Inkling 2's answer format."""
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

tok = Tokenizer.from_file(str(Path(sys.argv[1]) / "tokenizer.json"))
E = "<|end_message|>"
answers = [
    ["22C and sunny", "<|message_model|><|content_thinking|>The tool says sunny." + E +
                      "<|message_model|><|content_text|>It is 22C and sunny in Paris." + E + "<|content_model_end_sampling|>"],
    ["weather", "<|message_model|><|content_thinking|>I should call the weather tool." + E +
                '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris","unit":"C"}}' + E +
                "<|content_model_end_sampling|>"],
    ["count to", "<|message_model|><|content_text|>one two three STOP four five" + E + "<|content_model_end_sampling|>"],
    ["", "<|message_model|><|content_thinking|>The user greets me. 你好 needs a reply." + E +
         "<|message_model|><|content_text|>Hello! 你好，有什么可以帮你？" + E + "<|content_model_end_sampling|>"],
]
out = [[m, tok.encode(t, add_special_tokens=False).ids] for m, t in answers]
for (m, t), (_, ids) in zip(answers, out):
    assert tok.decode(ids, skip_special_tokens=False) == t, (t, tok.decode(ids, skip_special_tokens=False))
Path(sys.argv[2]).write_text(json.dumps(out))
print("script:", [(m, len(ids)) for m, ids in out])
