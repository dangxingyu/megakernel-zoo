"""The structure of a model's answer, read as it streams: reasoning, text and tool calls.

A checkpoint that describes its own answer format in tokenizer_config.json (`response_template`, as Inkling 2 does)
is parsed with that description: each field has an opening pattern (a regular expression over the decoded text,
special tokens included), a closing string, and a kind (text, streamed as it arrives; json, parsed when it closes and
turned into an object by `transform`). Inkling 2's answer, for example:

    <|message_model|><|content_thinking|>...<|end_message|>                 reasoning_content
    <|message_model|><|content_text|>...<|end_message|>                     content
    <|message_model|>get_weather<|content_invoke_tool_json|>{"name":..., "args":{...}}<|end_message|>   tool_calls

A checkpoint without one is answered as plain text (special tokens left out), as before.
"""

from __future__ import annotations

import json
import re

SPECIAL = re.compile(r"<\|[A-Za-z0-9_]+\|>")


def load_response_template(model_dir) -> dict | None:
    from pathlib import Path
    path = Path(model_dir) / "tokenizer_config.json"
    if not path.exists():
        return None
    rt = json.loads(path.read_text()).get("response_template")
    return rt if isinstance(rt, dict) and rt.get("fields") else None


def _fill(spec, content):
    """`transform`: strings "{content.KEY}" are replaced by that value of the parsed JSON (not stringified)."""
    if isinstance(spec, str):
        m = re.fullmatch(r"\{content\.([A-Za-z0-9_]+)\}", spec)
        return content.get(m.group(1)) if m and isinstance(content, dict) else spec
    if isinstance(spec, dict):
        return {k: _fill(v, content) for k, v in spec.items()}
    if isinstance(spec, list):
        return [_fill(v, content) for v in spec]
    return spec


class ResponseParser:
    """feed(text) -> [(field, value)]: value is a text piece for text fields and a finished object for json fields.
    Text outside every field (no opening marker) is answered as `content`, without special tokens."""

    def __init__(self, rt: dict):
        self.fields = []
        for name, f in rt["fields"].items():
            self.fields.append({"name": name, "open": re.compile(f["open_pattern"]), "close": f["close"],
                                "kind": f.get("content", "text"), "transform": f.get("transform")})
        self.buf = ""
        self.cur = None
        self.acc = ""

    def _stray(self, text: str, out: list) -> None:
        text = SPECIAL.sub("", text)
        if text:
            out.append(("content", text))

    def feed(self, text: str) -> list:
        out: list = []
        self.buf += text
        while self.buf:
            if self.cur is None:
                best = None
                for f in self.fields:
                    m = f["open"].search(self.buf)
                    if m and (best is None or m.start() < best[1].start()):
                        best = (f, m)
                if best is None:
                    # keep what may still become an opening marker (from the last special token on), let go of the rest
                    i = self.buf.rfind("<|")
                    keep = self.buf[i:] if i >= 0 else ""
                    self._stray(self.buf[: len(self.buf) - len(keep)], out)
                    self.buf = keep
                    break
                f, m = best
                self._stray(self.buf[: m.start()], out)
                self.cur, self.acc, self.buf = f, "", self.buf[m.end():]
                continue
            close = self.cur["close"]
            i = self.buf.find(close)
            if i < 0:
                keep = next((k for k in range(min(len(close) - 1, len(self.buf)), 0, -1) if close.startswith(self.buf[-k:])), 0)
                piece, self.buf = self.buf[: len(self.buf) - keep], self.buf[len(self.buf) - keep:]
                self._piece(piece, out)
                break
            piece, self.buf = self.buf[:i], self.buf[i + len(close):]
            self._piece(piece, out)
            self._finish(out)
        return out

    def _piece(self, piece: str, out: list) -> None:
        if not piece:
            return
        if self.cur["kind"] == "text":
            out.append((self.cur["name"], piece))
        else:
            self.acc += piece

    def _finish(self, out: list) -> None:
        f, self.cur = self.cur, None
        if f["kind"] == "text":
            return
        try:
            obj = json.loads(self.acc)
        except json.JSONDecodeError:  # not valid JSON: show it rather than lose it
            out.append(("content", self.acc))
            return
        out.append((f["name"], _fill(f["transform"], obj) if f.get("transform") else obj))

    def close(self) -> list:
        """The answer ended (end of sequence, a stop, max_tokens): whatever is open is closed."""
        out: list = []
        if self.cur is not None:
            self._piece(self.buf, out)
            self.buf = ""
            if self.cur["kind"] == "text":
                self.cur = None
            else:
                self._finish(out)
        elif self.buf:
            self._stray(self.buf, out)
            self.buf = ""
        return out
