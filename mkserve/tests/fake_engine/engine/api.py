"""A stand-in engine with the online contract (submit / poll / cancel) for testing mkserve's APIs without a GPU.
MKSERVE_FAKE_SCRIPT: a JSON list of [marker, token ids]; a request is answered with the first entry whose marker
occurs in the last 400 characters of its decoded prompt, three tokens per poll as a speculative engine would. Each
prompt is appended to MKSERVE_FAKE_LOG (decoded, special tokens kept)."""
import json
import os
from pathlib import Path

from tokenizers import Tokenizer


class Engine:
    def __init__(self, model_dir, max_batch, max_context):
        self.script = json.load(open(os.environ["MKSERVE_FAKE_SCRIPT"]))
        self.log = os.environ.get("MKSERVE_FAKE_LOG")
        self.tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.active = []

    def submit(self, rid, prompt, n):
        text = self.tok.decode(list(prompt), skip_special_tokens=False)
        reply = next(toks for marker, toks in self.script if marker in text[-400:])
        if self.log:
            with open(self.log, "a") as f:
                f.write(json.dumps({"rid": rid, "n": n, "prompt": text}) + "\n")
        self.active.append({"rid": rid, "left": list(reply)[:n] + [7] * max(0, n - len(reply))})

    def cancel(self, rid):
        self.active = [a for a in self.active if a["rid"] != rid]

    def poll(self):
        events = []
        for a in list(self.active):
            toks, a["left"] = a["left"][:3], a["left"][3:]
            done = not a["left"]
            events.append((a["rid"], toks, done))
            if done:
                self.active.remove(a)
        return events
