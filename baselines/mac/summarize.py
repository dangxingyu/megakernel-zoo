#!/usr/bin/env python
"""python baselines/mac/summarize.py OUT.json RUN_DIR...: one baselines file (eval/baselines_<model>_<card>_c.json shape)
from bench.sh result directories.

Per system and workload, the median over server starts of each start's median repeat:
  synth, natural  decode_tok_s_by_tokens (1,024 new tokens: tokens arriving while the stream decodes / that window,
                  every chunk weighted by the tokens it carries)
  chat            (tokens - 1) / (last token time - first token time), summed over the repeats of a start: the decode
                  rate after the first token, from the client's own clock and the server's token count. The chat
                  streams end at the model's own end token (no ignore_eos), so they vary in length (up to 512). Not the
                  token-based decode rate: oMLX streams a chunk every ~128 ms with 10-15 tokens and no per-chunk counts,
                  and on a short stream the even split over chunks overstates the rate (by 40% at 256 tokens). Not the
                  end-to-end rate: the prefill differs by system and is reported as ttft (oMLX's first SSE chunk is an
                  empty string, so its client-side ttft reads ~0.01 s; its own log says 0.06-0.08 s to the first token).
Values are microseconds per token (1e6 / tok/s) like the B200 file, plus the rates and tokens per chunk."""
import json
import statistics as st
import sys
from pathlib import Path

out, dirs = Path(sys.argv[1]), [Path(d) for d in sys.argv[2:]]
rows: dict[str, dict[str, list[dict]]] = {}
for d in dirs:
    for f in sorted(d.glob("*-r*-*.json")):
        sysname, rest = f.stem.rsplit("-r", 1)
        rnd, workload = rest.split("-", 1)
        res = json.loads(f.read_text())
        if not res.get("clean_repeats"):
            continue
        rows.setdefault(sysname, {}).setdefault(workload, []).append(res)

key = {"synth": ("decode_tok_s_by_tokens", "b1"), "natural": ("decode_tok_s_by_tokens", "b1_natural"),
       "chat": ("e2e_tok_s", "b1_chat")}
systems = {}
for sysname, by_w in sorted(rows.items()):
    entry = {"tok_s": {}, "tokens_per_chunk": {}, "per_start_tok_s": {}, "starts": {}}
    for w, results in by_w.items():
        metric, name = key[w]
        if w == "chat":
            per_start = []
            for r in results:
                reps = [x for x in r.get("per_repeat", []) if x.get("clean")]
                toks = sum(sum(x["tokens"]) - len(x["tokens"]) for x in reps)
                span = sum(x["wall_s"] - max(x["ttft_s"]) for x in reps)
                if span > 0:
                    per_start.append(toks / span)
            entry.setdefault("ttft_s", {})[w] = round(st.median(t for r in results for x in r.get("per_repeat", [])
                                                                if x.get("clean") for t in x["ttft_s"]), 3)
            entry.setdefault("tokens_per_stream", {})[w] = st.median(r["tokens_per_stream"] for r in results)
        else:
            per_start = [r[metric]["median"] for r in results if isinstance(r.get(metric), dict)]
        if not per_start:
            continue
        rate = st.median(per_start)
        entry[name] = round(1e6 / rate)
        entry["tok_s"][w] = round(rate, 1)
        entry["per_start_tok_s"][w] = [round(v, 1) for v in per_start]
        entry["tokens_per_chunk"][w] = round(st.median(r["tokens_per_chunk"] for r in results), 2)
        entry["starts"][w] = len(per_start)
    systems[sysname] = entry

machine = (dirs[0] / "machine.txt").read_text().split("\n")[:2] if (dirs[0] / "machine.txt").exists() else []
doc = {"model": "thinkingmachines/Inkling-2-Mini", "gpu": "Apple M5 Max (40-core GPU, 48 GB)", "hw": "m5max", "track": "c",
       "provisional": True, "machine": machine,
       "calibre": "server end to end over HTTP, client clock (eval/client.py steady through baselines/mac/bench.sh), B=1, "
                  "greedy, ignore_eos, text prompts; 1 warm-up + 3 repeats per server start, 2 starts per system in turns. "
                  "b1: synthetic 64-token prompts, 1,024 new tokens, token-based decode rate; b1_natural: "
                  "eval/natural_prompts.py, same rate; b1_chat: GSM8K through the chat template, 256 new tokens, "
                  "end-to-end rate (see baselines/mac/summarize.py). Microseconds per token.",
       "systems": systems}
out.write_text(json.dumps(doc, indent=1) + "\n")
for s, e in systems.items():
    print(f"{s:18s}", "  ".join(f"{w}={e['tok_s'].get(w, float('nan')):6.1f} tok/s" for w in ("synth", "natural", "chat")))
