#!/usr/bin/env python
"""python baselines/mac/make_prompts.py MODEL_DIR OUT_DIR: the three bs=1 workloads' prompt files for this Mac.

  natural.json  eval/natural_prompts.py: 48 prompts of 64 tokens (prose, Python source, GSM8K), token ids
  natural_text.json  the same decoded to text (servers that take no token ids)
  chat.json     48 GSM8K test questions as chat messages (strings; eval/client.py --api chat --prompts-file),
                the track-C baselines' "chat" workload: the model answers through its own chat template
The synthetic workload needs no file (eval/client.py makes its prompts from the seed)."""
import json
import random
import subprocess
import sys
import urllib.request
from pathlib import Path

model_dir, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
zoo = Path(__file__).resolve().parents[2]
subprocess.run([sys.executable, str(zoo / "eval" / "natural_prompts.py"), model_dir, str(out / "natural.json"), "64", "48"],
               check=True)
from tokenizers import Tokenizer
tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
ids = json.loads((out / "natural.json").read_text())
text = [tok.decode(x) for x in ids]
same = sum(tok.encode(t, add_special_tokens=False).ids == x for t, x in zip(text, ids))
(out / "natural_text.json").write_text(json.dumps(text))   # for servers that take text only; check how many encode back
print(f"MK_ENV natural_text: {same}/{len(ids)} encode back to the same ids")
url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
rows = [json.loads(line) for line in urllib.request.urlopen(url, timeout=60).read().decode().splitlines() if line.strip()]
random.Random(20261122).shuffle(rows)
(out / "chat.json").write_text(json.dumps([r["question"] for r in rows[:48]], indent=0))
print(f"MK_ENV prompts natural=48 chat=48 (gsm8k test, seed 20261122) -> {out}")
