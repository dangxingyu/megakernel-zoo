"""Prompts with an exact token count, different for every (length, seed).

A prompt is a header that is unique to its seed (so no prefix cache can reuse an earlier prompt), filler
words, and a fixed closing instruction that invites a long answer. `ids` has exactly `n` tokens under the
model's tokenizer. Servers that accept token ids get `ids`; text-only servers get `text`, whose token count
`exact_text` checks by encoding it again.
"""

from __future__ import annotations

import random
from pathlib import Path

from tokenizers import Tokenizer

WORDS = ("the of and to in is that it for was on are as with his they at be this from have or by one had not but "
         "what all were when we there can an your which their said if do will each about how up out them then she "
         "many some so these would other into has more her two like him see time could no make than first been its "
         "who now people my made over did down only way find use may water long little very after words called "
         "just where most know river mountain garden letter window market engine signal number system").split()
TAIL = "\n\nContinue the text above with a very long and detailed passage:\n"


def load(model_dir: str) -> Tokenizer:
    return Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))


def make(tok: Tokenizer, n: int, seed: int) -> dict:
    """{"ids": n token ids, "text": their decoding, "exact_text": whether the text encodes back to n tokens}."""
    rng = random.Random(seed * 1_000_003 + n)
    tail = tok.encode(TAIL, add_special_tokens=False).ids
    head = tok.encode(f"Document {seed:08x}-{rng.getrandbits(48):012x}.", add_special_tokens=False).ids
    if n <= len(tail) + len(head):
        ids = (head + tail)[:n]
    else:
        body: list[int] = []
        need = n - len(head) - len(tail)
        while len(body) < need:
            chunk = " " + " ".join(rng.choice(WORDS) for _ in range(256))
            body += tok.encode(chunk, add_special_tokens=False).ids
        ids = head + body[:need] + tail
    text = tok.decode(ids, skip_special_tokens=False)
    return {"ids": ids, "text": text,
            "exact_text": len(tok.encode(text, add_special_tokens=False).ids) == len(ids)}


def make_with_prefix(tok: Tokenizer, prefix_n: int, suffix_n: int, seed: int, prefix_seed: int = 424_242) -> dict:
    """A prompt of prefix_n + suffix_n tokens whose first prefix_n tokens are the same for every seed (a shared
    system prompt or document: what a prefix cache can reuse) and whose last suffix_n tokens are unique to the seed."""
    suffix = make(tok, suffix_n, seed)["ids"]
    if prefix_n <= 0:
        ids = suffix
    else:
        rng = random.Random(prefix_seed)
        prefix: list[int] = tok.encode("Shared reference text.", add_special_tokens=False).ids
        while len(prefix) < prefix_n:
            prefix += tok.encode(" " + " ".join(rng.choice(WORDS) for _ in range(256)), add_special_tokens=False).ids
        ids = prefix[:prefix_n] + suffix
    text = tok.decode(ids, skip_special_tokens=False)
    return {"ids": ids, "text": text, "exact_text": len(tok.encode(text, add_special_tokens=False).ids) == len(ids)}


def word_pool(tok: Tokenizer) -> list[int]:
    """Token ids of common words that are one token each with a leading space: building blocks of exact length."""
    pool = []
    for word in WORDS:
        ids = tok.encode(" " + word, add_special_tokens=False).ids
        if len(ids) == 1:
            pool.append(ids[0])
    return pool


def from_block_hashes(pool: list[int], hash_ids: list[int], length: int, block: int = 512) -> list[int]:
    """Token ids for a trace record that gives its prompt as a list of prefix-block hashes (Mooncake traces):
    block k is `block` tokens drawn from a generator seeded by hash_ids[k], so two prompts share exactly the
    leading tokens whose hashes they share, which is what a prefix cache can reuse."""
    ids: list[int] = []
    for h in hash_ids:
        rng = random.Random(h * 2_654_435_761 + 17)
        ids += [pool[rng.randrange(len(pool))] for _ in range(block)]
        if len(ids) >= length:
            break
    while len(ids) < length:  # a record whose length exceeds its blocks: pad with content unique to its last hash
        rng = random.Random((hash_ids[-1] if hash_ids else 0) * 97 + len(ids))
        ids.append(pool[rng.randrange(len(pool))])
    return ids[:length]
