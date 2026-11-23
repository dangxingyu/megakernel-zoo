#!/usr/bin/env python
"""The headline of a sealed evaluation, in the calibre of the baselines: the engine behind an HTTP server, measured by
the same client that measures vLLM, SGLang and Cohere's megakernel (scripts/compare.sh does the same by hand).

    python eval/http_stage.py --workspace WS --model-dir D --out RESULT.json [--max-tokens 1024] [--repeats 3]

What it does, on the card the evaluation has:
  1. starts `eval/serve_engine.py engine` (the engine's process, the evaluation venv, cwd = WS) and
     `eval/serve_engine.py frontend` (HTTP, tokenizer, SSE) on the server cores;
  2. checks the tokens the server returns: the reference prompts of the correctness check, all at once, must give the
     tokens the engine's own step-by-step greedy decode gave in that check (`expect`), with the rule of the `generate`
     gate (they may differ only at exact ties). Without this the headline would be measured on a path nobody checked;
  3. runs `eval/client.py steady` on the client cores: prompt 64, 1024 new tokens, greedy, one SSE chunk per token,
     batch 1 and batch 8, one warm-up and three repeats; time per step = 1e6 * B / (decode tokens per second).

The server uses the engine's submit()/poll() when it has them (an engine that keeps its loop on the device can then
still stream), else step() through a cohort adapter. Standard library only in this file; the client and the front end
run under the tools Python (aiohttp, tokenizers).

Track C adds `sequential_check`: the same reference prompts again, one request at a time (`tokens_b1`), so that the
single-sequence path (where a speculative engine drafts and verifies) is checked too. On Apple silicon (macOS) the
client samples no NVML, and kernel_eval.py passes batches=(1,) with the all-at-once check off unless B=8 is asked for.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def pin(cpus: str | None) -> list[str]:
    return ["taskset", "-c", cpus] if cpus and shutil.which("taskset") else []


def kill(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=8)
            return
        except subprocess.TimeoutExpired:
            continue


def tail(path: Path, limit: int = 1500) -> str:
    try:
        return path.read_text(errors="replace")[-limit:]
    except OSError:
        return ""


def stream_ids(port: int, ids: list[int], n: int, timeout: float) -> list[int]:
    """One streaming completion; the token ids the server says it generated."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    body = json.dumps({"model": "m", "prompt": ids, "max_tokens": n, "stream": True, "ignore_eos": True, "mk_return_ids": True})
    conn.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    out: list[int] = []
    for raw in resp:
        line = raw.strip()
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        for choice in json.loads(line[6:]).get("choices") or []:
            out += choice.get("mk_token_ids") or []
    conn.close()
    return out


def burst_stats(port: int, reset: bool = False) -> dict:
    """The front end's tokens-per-engine-event counter (eval/serve_engine.py /mk/bursts); {} if unavailable."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/mk/bursts{'?reset=1' if reset else ''}", timeout=5) as resp:
            return json.loads(resp.read())
    except (OSError, ValueError):
        return {}


def check_tokens(port: int, prompts: list[list[int]], expect: list[list[int]], timeout: float) -> dict:
    got: list = [None] * len(prompts)
    errors: list[str] = []

    def one(i: int) -> None:
        try:
            got[i] = stream_ids(port, prompts[i], len(expect[i]), timeout)
        except Exception as error:
            errors.append(f"{type(error).__name__}: {str(error)[:200]}")

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors or any(g is None for g in got):
        return {"pass": False, "errors": errors[:3]}
    prefix, identical, lengths = [], 0, True
    for g, e in zip(got, expect):
        lengths = lengths and len(g) == len(e)
        same = 0
        for a, b in zip(g, e):
            if a != b:
                break
            same += 1
        prefix.append(same / max(1, len(e)))
        identical += int(g == e)
    mean = sum(prefix) / len(prefix)
    return {"pass": bool(lengths and mean >= 0.75 and identical * 2 >= len(expect)), "mean_matching_prefix": mean,
            "sequences_identical": identical, "of": len(expect), "lengths_ok": lengths,
            "note": "tokens streamed by the HTTP server for the reference prompts, all at once, against the engine's own step-by-step greedy tokens"}


def check_tokens_sequential(port: int, prompts: list[list[int]], expect: list[list[int]], timeout: float) -> dict:
    """The check of check_tokens with one request at a time: each prompt alone on the server (batch 1), the next one
    sent when the previous one has finished."""
    got: list = []
    errors: list[str] = []
    for p, e in zip(prompts, expect):
        try:
            got.append(stream_ids(port, p, len(e), timeout))
        except Exception as error:
            errors.append(f"{type(error).__name__}: {str(error)[:200]}")
            break
    if errors or len(got) != len(expect):
        return {"pass": False, "errors": errors[:3]}
    prefix, identical, lengths = [], 0, True
    for g, e in zip(got, expect):
        lengths = lengths and len(g) == len(e)
        same = next((i for i, (a, b) in enumerate(zip(g, e)) if a != b), min(len(g), len(e)))
        prefix.append(same / max(1, len(e)))
        identical += int(g == e)
    mean = sum(prefix) / len(prefix)
    return {"pass": bool(lengths and mean >= 0.75 and identical * 2 >= len(expect)), "mean_matching_prefix": mean,
            "sequences_identical": identical, "of": len(expect), "lengths_ok": lengths,
            "first_difference": [next((i for i, (a, b) in enumerate(zip(g, e)) if a != b), None) for g, e in zip(got, expect)],
            "note": "tokens streamed by the HTTP server for the reference prompts, one request at a time (batch 1), against "
                    "the engine's own step-by-step greedy tokens"}


def measure(workspace: str, model_dir: str, eval_py: str, tools_py: str, scratch: Path, server_cpus: str | None = None,
            client_cpus: str | None = None, batches=(1, 8), max_tokens: int = 1024, repeats: int = 3, warmup: int = 1,
            prompt_tokens: int = 64, startup_s: float = 1800.0, run_s: float = 3600.0, read_s: float = 300.0, prompts: list | None = None,
            expect: list | None = None, max_batch: int = 8, max_context: int = 2048, rate: str = "chunks",
            natural_prompts: str | None = None, concurrent_check: bool = True, sequential_check: bool = False) -> dict:
    """rate "chunks": time per step = 1e6 * B / decode_tok_s (one SSE chunk per token, tracks A and B); "tokens" (track
    C): 1e6 * B / decode_tok_s_by_tokens, every chunk weighted by the tokens it carries (eval/client.py chunk_weights;
    the same number when every chunk carries one token). natural_prompts: a prompts file (eval/natural_prompts.py)
    measured as a second workload, reported as natural_b1 / natural_b8. concurrent_check: the token check with every
    prompt sent at once (`tokens`); sequential_check: the same prompts one request at a time (`tokens_b1`)."""
    scratch.mkdir(parents=True, exist_ok=True)
    sock, port = scratch / "http.sock", free_port()
    sock.unlink(missing_ok=True)
    logs = {"engine": scratch / "http-engine.log", "frontend": scratch / "http-frontend.log"}
    env = dict(os.environ, PATH=f"{Path(eval_py).parent}:{os.environ.get('PATH', '')}")
    out: dict = {"calibre": "server end to end over HTTP: 1e6 * B / decode tokens per second, on the client's clock "
                            f"(prompt {prompt_tokens}, {max_tokens} new tokens, greedy, one SSE chunk per token, {warmup} warm-up + {repeats} repeats); "
                            "the same client, workload and formula as the baselines",
                 "server_cpus": server_cpus, "client_cpus": client_cpus, "ok": False}
    engine = frontend = None
    t0 = time.time()
    try:
        engine = subprocess.Popen(pin(server_cpus) + [eval_py, str(HERE / "serve_engine.py"), "engine", "--workspace", workspace, "--model-dir", model_dir,
                                                      "--sock", str(sock), "--max-batch", str(max_batch), "--max-context", str(max_context)],
                                  cwd=workspace, env=env, stdout=open(logs["engine"], "wb"), stderr=subprocess.STDOUT, start_new_session=True)
        frontend = subprocess.Popen(pin(server_cpus) + [tools_py, str(HERE / "serve_engine.py"), "frontend", "--model-dir", model_dir, "--sock", str(sock),
                                                        "--port", str(port), "--wait-s", str(startup_s)],
                                    stdout=open(logs["frontend"], "wb"), stderr=subprocess.STDOUT, start_new_session=True)
        up = False
        while time.time() - t0 < startup_s and engine.poll() is None and frontend.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as resp:
                    up = resp.status == 200
            except OSError:
                up = False
            if up:
                break
            time.sleep(1.0)
        out["startup_seconds"] = round(time.time() - t0, 1)
        text = tail(logs["engine"], 4000)
        out["contract"] = "online (submit/poll)" if "contract=online" in text else "offline (step through the cohort adapter)" if "contract=offline" in text else None
        if not up:
            out["error"] = "the HTTP server did not come up"
            out["engine_log"], out["frontend_log"] = tail(logs["engine"]), tail(logs["frontend"])
            return out
        if prompts and expect and concurrent_check:
            # a server that sends nothing for read_s seconds counts as hung (the engine's first token of a request
            # needs the prompt's steps, nothing more)
            out["tokens"] = check_tokens(port, prompts, expect, timeout=read_s)
            out["tokens_seconds"] = round(time.time() - t0 - out["startup_seconds"], 1)
            if engine.poll() is not None or frontend.poll() is not None or out["tokens"].get("errors"):
                out["error"] = "the server died or hung during the token check (the check's eight prompts sent at once)"
                out["engine_log"], out["frontend_log"] = tail(logs["engine"]), tail(logs["frontend"])
                return out
        if prompts and expect and sequential_check:
            t1 = time.time()
            out["tokens_b1"] = check_tokens_sequential(port, prompts, expect, timeout=read_s)
            out["tokens_b1_seconds"] = round(time.time() - t1, 1)
            if engine.poll() is not None or frontend.poll() is not None or out["tokens_b1"].get("errors"):
                out["error"] = "the server died or hung during the one-request-at-a-time token check"
                out["engine_log"], out["frontend_log"] = tail(logs["engine"]), tail(logs["frontend"])
                return out
        out["rate"] = rate
        workloads = [("", None)] + ([("natural_", natural_prompts)] if natural_prompts else [])
        for (wname, pfile), batch in [(w, b) for w in workloads for b in batches]:
            result_path = scratch / f"http-{wname}b{batch}.json"
            cmd = pin(client_cpus) + [tools_py, str(HERE / "client.py"), "steady", "--base", f"http://127.0.0.1:{port}", "--model", "m", "--model-dir", model_dir,
                                      "--batch", str(batch), "--prompt-tokens", str(prompt_tokens), "--max-tokens", str(max_tokens), "--repeats", str(repeats),
                                      "--warmup", str(warmup)] + (["--nvml"] if platform.system() != "Darwin" else []) + [
                                      "--read-timeout", str(read_s), "--extra-body", '{"ignore_eos": true}',
                                      "--label", "eval-http", "--out", str(result_path)] + (["--prompts-file", pfile] if pfile else [])
            burst_stats(port, reset=True)
            try:
                done = subprocess.run(cmd, capture_output=True, text=True, timeout=run_s)
                rc, text = done.returncode, (done.stdout + done.stderr)[-1500:]
            except subprocess.TimeoutExpired:
                rc, text = 124, f"the client did not finish in {run_s:.0f} s"
            try:
                result = json.loads(result_path.read_text())
            except (OSError, ValueError):
                result = None
            tok = (result or {}).get("decode_tok_s_by_tokens" if rate == "tokens" else "decode_tok_s") or {}
            key = f"{wname}b{batch}"
            if not result or not tok.get("median") or not result.get("clean_repeats"):
                out[key] = {"error": f"no clean measurement (client rc {rc})", "client_tail": text}
                out["error"] = f"no clean HTTP measurement at batch {batch}{' (natural text)' if wname else ''} (a server that sends nothing for {read_s:.0f} s counts as hung)"
                out["engine_log"], out["frontend_log"] = tail(logs["engine"]), tail(logs["frontend"])
                return out
            chunk_tok = result.get("decode_tok_s") or {}
            out[key] = {"step_us": 1e6 * batch / tok["median"], "step_us_min": 1e6 * batch / tok["max"], "step_us_max": 1e6 * batch / tok["min"],
                                "rate": rate, "decode_tok_s": tok, "decode_tok_s_chunks": chunk_tok,
                                "step_us_chunks": 1e6 * batch / chunk_tok["median"] if chunk_tok.get("median") else None,
                                "tokens_per_chunk": result.get("tokens_per_chunk"), "engine_tokens_per_event": burst_stats(port).get("tokens_per_event"),
                                "engine_events": burst_stats(port).get("hist"),
                                "e2e_tok_s": result.get("e2e_tok_s"), "repeats": result.get("repeats"),
                                "clean_repeats": result.get("clean_repeats"), "tokens_per_stream": result.get("tokens_per_stream"),
                                "one_chunk_per_token": result.get("one_chunk_per_token"), "tpot_p50_ms": result.get("tpot_p50_ms"),
                                "tpot_p99_ms": result.get("tpot_p99_ms"), "ttft_median_s": result.get("ttft_median_s"),
                                "nvml_util_mean": result.get("nvml_util_mean"),
                                "nvml_util_per_card": result.get("nvml_util_per_card"), "client_rc": rc}
            if engine.poll() is not None or frontend.poll() is not None:
                out["error"] = f"the server died during the batch-{batch} measurement"
                out["engine_log"] = tail(logs["engine"])
                return out
        out["ok"] = all(f"{w}b{b}" in out and "step_us" in out[f"{w}b{b}"] for w, _ in workloads for b in batches) and \
            all(out[k]["pass"] for k in ("tokens", "tokens_b1") if prompts and expect and k in out)
        return out
    finally:
        kill(frontend)
        kill(engine)
        out["seconds"] = round(time.time() - t0, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--model-dir", default=os.environ.get("MK_MODEL_DIR"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--python", default=os.environ.get("MK_EVAL_PY") or sys.executable)
    parser.add_argument("--tools-python", default=os.environ.get("MK_TOOLS_PY") or sys.executable)
    parser.add_argument("--scratch", default=os.environ.get("MK_EVAL_SCRATCH") or ("/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()))
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batches", default=os.environ.get("MK_EVAL_BATCHES") or ("1" if platform.system() == "Darwin" else "1,8"))
    args = parser.parse_args()
    scratch = Path(args.scratch) / f"mk-http-{os.getpid()}"
    batches = [int(b) for b in args.batches.split(",")]
    result = measure(str(Path(args.workspace).resolve()), args.model_dir, args.python, args.tools_python, scratch,
                     os.environ.get("MK_SERVER_CPUS"), os.environ.get("MK_CLIENT_CPUS"), batches,
                     args.max_tokens, args.repeats, args.warmup, max_batch=max(batches) if platform.system() == "Darwin" else 8,
                     concurrent_check=True, sequential_check=False)
    shutil.rmtree(scratch, ignore_errors=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print("MK_HTTP " + json.dumps(result)[:3000], flush=True)
    return 0 if result.get("ok") else 3


if __name__ == "__main__":
    raise SystemExit(main())
