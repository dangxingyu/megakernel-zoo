#!/usr/bin/env python
"""python eval/natural_prompts.py MODEL_DIR OUT.json [TOKENS=64] [COUNT=48]: prompts of natural text, each exactly TOKENS
tokens of this model's tokenizer, as a JSON list of token-id lists (eval/client.py --prompts-file).

A third each of English prose (public-domain books from Project Gutenberg), Python source (the standard library of the
pod's own interpreter) and word problems (the GSM8K test set), interleaved so that any few consecutive prompts mix the
three. Passages are cut at TOKENS tokens: engines that batch only prompts of equal length then decode them together.
Without network access the prose falls back to the interpreter's pydoc topics and the questions to more source code.
Prints one MK_ENV line saying where the text came from. The same seed gives the same prompts for the same tokenizer."""
import json
import random
import sys
import sysconfig
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

model_dir, out = sys.argv[1], sys.argv[2]
n_tok = int(sys.argv[3]) if len(sys.argv) > 3 else 64
count = int(sys.argv[4]) if len(sys.argv) > 4 else 48
tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
rng = random.Random(int(__import__("os").environ.get("MK_NATURAL_SEED") or 20261115))  # suite v2: the hidden set uses its own seed


def fetch(url):
    try:
        return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "prompt-builder"}), timeout=60).read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None


def cut(text):
    ids = tok.encode(text, add_special_tokens=False).ids
    return ids[:n_tok] if len(ids) >= n_tok else None


def passages(text, k, min_chars=700):
    """k passages that start at a paragraph break, spread over the text."""
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 200]
    out = []
    for i in rng.sample(range(len(paras)), min(len(paras), 4 * k)):
        ids = cut(" ".join(" ".join(paras[i:i + 3]).split())[:6000])
        if ids:
            out.append(ids)
        if len(out) == k:
            break
    return out


per, src = (count + 2) // 3, {}
prose = []
for book in (1342, 2701, 11, 84):  # Pride and Prejudice, Moby Dick, Alice in Wonderland, Frankenstein
    # the same file from Project Gutenberg's own mirror when gutenberg.org rate-limits (HTTP 503)
    text = (fetch(f"https://www.gutenberg.org/cache/epub/{book}/pg{book}.txt")
            or (fetch(f"https://gutenberg.pglaf.org/cache/epub/{book}/pg{book}.txt") or "").replace("\r\n", "\n"))
    if text and len(text) > 50000:
        body = text[len(text) // 10: -len(text) // 10]  # away from the licence text at both ends
        prose += passages(body, (per + 3) // 4)
src["prose"] = "Project Gutenberg" if len(prose) >= per else "pydoc topics"
if len(prose) < per:
    import pydoc_data.topics as topics
    prose += passages("\n\n".join(topics.topics.values()), per - len(prose))
code = []
files = sorted(Path(sysconfig.get_paths()["stdlib"]).glob("*.py"))
for f in rng.sample(files, len(files)):
    text = f.read_text(errors="replace")
    at = text.find("\ndef ", len(text) // 4)
    ids = cut(text[at + 1:at + 6000]) if at > 0 else None
    if ids:
        code.append(ids)
    if len(code) >= 2 * per:
        break
questions = []
gsm = fetch("https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl")
if gsm:
    for line in gsm.splitlines():
        if line.strip():
            ids = cut(json.loads(line)["question"])
            if ids:
                questions.append(ids)
        if len(questions) >= per:
            break
src["questions"] = "GSM8K test" if len(questions) >= per else "more source code"
if len(questions) < per:
    questions += code[per:per + per - len(questions)]
kinds = [prose[:per], code[:per], questions[:per]]
prompts = [k[i] for i in range(per) for k in kinds if i < len(k)][:count]
assert len(prompts) >= 16 and all(len(p) == n_tok for p in prompts), (len(prompts), n_tok)
Path(out).write_text(json.dumps(prompts))
print(f"MK_ENV natural prompts: {len(prompts)} of {n_tok} tokens; prose from {src['prose']}, code from the Python standard library, questions from {src['questions']}")
