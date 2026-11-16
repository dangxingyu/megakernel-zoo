#!/usr/bin/env python
"""Evaluate one engine: build it, check it against the float64 reference, time it, trace it.

    python eval/kernel_eval.py --workspace WS --out RESULT.json [--quick] [--no-trace]

WS/engine/ holds the code under test (contract: eval/_engine_runner.py). Model, reference data and reference
lines come from the environment: MK_MODEL_DIR, MK_REF_DIR, MK_BASELINES (a JSON file, optional), and the card from
MK_HW (h100 | b200; default: what the baselines file says, else h100). The card decides the memory bandwidth of
the MBU figure (eval/modelinfo.py CARDS; MK_HBM_GB_S overrides it). "float64" below is the reference's precision:
float32 for a model whose float64 copy does not fit the card (meta.json and result.reference say which).

The engine runs in child processes; this process only compares and judges. Order: build, check, time, trace.
A stage that fails ends the evaluation with what is known so far.

The headline is measured in the calibre of the baselines (eval/http_stage.py): the engine behind an HTTP server, the
baselines' client and workload, time per step = 1e6 * B / decode tokens per second. Speedups and the milestones
against other systems use only that figure. Everything timed inside the engine's process (step, generate, trace, MBU)
is a diagnostic. `--no-http` (or MK_EVAL_HTTP=0) gives the older evaluation whose headline is the in-process
`step_us_b1`; a `--quick` evaluation has no HTTP stage either and is never ranked.

Result (the loop's contract; everything else is detail for the ledger and the figures):
  status     build_failed | error | incorrect | valid
  headline   {"metric": "http_step_us_b1", "value": ..., "lower_is_better": true}     only when valid
  gates      numeric (relative L2 error against float64, tolerance = 2x the Hugging Face BF16 path's own error),
             tokens (top-1 agreement with float64 under teacher forcing), finite, batch (the same sequence alone
             and in a batch), timed_steps (logits taken during the timed run are checked like any other)
  perf       per batch size: step_us (median of segments, min, max), tok_s, mbu; timing is "step, no sampling,
             host-driven, wall clock around N steps with a final synchronize"
  trace      per decode step at B=1, from CUPTI: kernel launches, host synchronizations, runtime API calls,
             memcpys, GPU idle share, idle gaps of 20 us or more (the GPU waiting for the host)
  milestones_passed
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
import subprocess
import sys
import time
from pathlib import Path

# In a pod prepared by scripts/eval_setup.sh, $MK_ROOT/eval_env.json says which Python has torch and where the
# reference is; started under another Python (the loop's worker has only the standard library), switch to it.
_SETUP = {}
_setup_file = Path(os.environ.get("MK_ROOT", "/tmp/mk")) / "eval_env.json"
if _setup_file.exists():
    _SETUP = json.loads(_setup_file.read_text()).get("private", {})
    if _SETUP.get("python") and os.path.realpath(sys.executable) != os.path.realpath(_SETUP["python"]) and not os.environ.get("MK_EVAL_REEXEC"):
        os.environ["MK_EVAL_REEXEC"] = "1"
        os.execv(_SETUP["python"], [_SETUP["python"], *sys.argv])

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import http_stage  # noqa: E402
import modelinfo  # noqa: E402

HBM_GB_S = 3350.0  # set in main() from the card: H100 80GB HBM3 peak, as in ForgeMegakernel, unless told otherwise
TOLERANCE = 2.0  # gate: error <= TOLERANCE x the Hugging Face BF16 path's own error against float64


def run(cmd: list[str], timeout: float, cwd: str | None = None, env: dict | None = None) -> dict:
    t0 = time.time()
    try:
        done = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        rc, text = done.returncode, done.stdout[-6000:] + done.stderr[-6000:]
    except subprocess.TimeoutExpired as error:
        rc, text = 124, f"timeout after {timeout:.0f}s\n" + str(error.stdout or "")[-3000:] + str(error.stderr or "")[-3000:]
    return {"rc": rc, "seconds": round(time.time() - t0, 1), "tail": text[-4000:]}


def rel_l2(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64); ref = ref.astype(np.float64)
    a -= a.mean(-1, keepdims=True); ref = ref - ref.mean(-1, keepdims=True)
    return np.linalg.norm(a - ref, axis=-1) / np.linalg.norm(ref, axis=-1)


def stats(x: np.ndarray) -> dict:
    flat = np.sort(x.reshape(-1))
    return {"median": float(flat[len(flat) // 2]), "p99": float(flat[int(0.99 * len(flat))]), "max": float(flat[-1])}


def decode_bytes(model_dir: str, batch: int, context: float) -> dict:
    """Bytes one decode step has to read (eval/modelinfo.py): every weight of the text decoder except the input
    embedding table, k of N experts per layer for a mixture-of-experts model (low and high at batch > 1), K and V
    of the earlier positions a layer can see, recurrent state. `total` (the low count) is what MBU uses."""
    return modelinfo.decode_bytes(model_dir, batch, context)


def card(baselines_path: str | None, setup: dict) -> dict:
    """Which card this evaluation runs on and its peak memory bandwidth: MK_HW, else the eval set-up, else the
    baselines file, else H100. MK_HBM_GB_S (or `hbm_gb_s` in the baselines file) overrides the table's figure."""
    lines = json.loads(Path(baselines_path).read_text()) if baselines_path and Path(baselines_path).exists() else {}
    hw = (os.environ.get("MK_HW") or setup.get("hw") or lines.get("hw") or "h100").lower()
    known = modelinfo.CARDS.get(hw, {})
    rate = float(os.environ.get("MK_HBM_GB_S") or lines.get("hbm_gb_s") or known.get("hbm_gb_s") or 3350.0)
    # Stage time limits were sized for Qwen3-8B; a model whose Hugging Face starter is several times slower per step
    # (many layers, an expert loop in Python, 80 GB to load) gets them multiplied.
    scale = float(os.environ.get("MK_EVAL_TIMEOUT_SCALE") or lines.get("eval_timeout_scale") or 1.0)
    return {"hw": hw, "hbm_gb_s": rate, "hbm_source": known.get("hbm_source", ""), "timeout_scale": scale}


def analyse_trace(sqlite_path: Path, steps: int, name: str = "mk_timed") -> dict:
    """CUPTI records inside one NVTX range of the traced run ("mk_timed": host-driven steps; "mk_generate": the
    engine's own loop), per decode step."""
    con = sqlite3.connect(sqlite_path)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    if "NVTX_EVENTS" not in tables or "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        return {"error": "trace has no NVTX or kernel table", "tables": sorted(t for t in tables if "CUPTI" in t or "NVTX" in t)}
    columns = [r[1] for r in con.execute("pragma table_info(NVTX_EVENTS)")]
    if "textId" in columns and "StringIds" in tables:
        rows = con.execute("select e.start, e.end from NVTX_EVENTS e left join StringIds s on e.textId = s.id "
                           "where coalesce(e.text, s.value) = ? and e.end is not null", (name,)).fetchall()
    else:
        rows = con.execute("select start, end from NVTX_EVENTS where text = ? and end is not null", (name,)).fetchall()
    if not rows:
        return {"error": f"NVTX range {name} not found"}
    w0, w1 = rows[-1]

    def count(table: str) -> int:
        return con.execute(f"select count(*) from {table} where start >= ? and start < ?", (w0, w1)).fetchone()[0] if table in tables else 0

    spans = con.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL where end > ? and start < ? order by start", (w0, w1)).fetchall()
    merged: list[list[int]] = []
    for a, b in spans:
        a, b = max(a, w0), min(b, w1)
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    busy = sum(b - a for a, b in merged)
    gaps = [(b[0] - a[1]) / 1e3 for a, b in zip(merged, merged[1:])]
    if merged:
        gaps += [(merged[0][0] - w0) / 1e3, (w1 - merged[-1][1]) / 1e3]
    long_gaps = [g for g in gaps if g >= 20.0]
    return {"window_ms": (w1 - w0) / 1e6, "steps": steps,
            "kernel_launches_per_step": len(spans) / steps,
            "host_syncs_per_step": count("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION") / steps,
            "runtime_calls_per_step": count("CUPTI_ACTIVITY_KIND_RUNTIME") / steps,
            "memcpys_per_step": count("CUPTI_ACTIVITY_KIND_MEMCPY") / steps,
            "gpu_idle_share": 1 - busy / (w1 - w0),
            "idle_us_per_step": (w1 - w0 - busy) / 1e3 / steps,
            "gaps_ge_20us_per_step": len(long_gaps) / steps,
            "gpu_waits_for_host_us_per_step": sum(long_gaps) / steps,
            "timing": "CUPTI activity inside the timed NVTX range, traced run (slower than the timed run)"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quick", action="store_true", help="fewer check sequences, one timing segment, no trace")
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--no-http", action="store_true", default=os.environ.get("MK_EVAL_HTTP") == "0",
                        help="no HTTP stage: the headline is the in-process step_us_b1 (the evaluation before 2026-11-10)")
    parser.add_argument("--tools-python", default=os.environ.get("MK_TOOLS_PY") or _SETUP.get("tools_python"))
    parser.add_argument("--model-dir", default=os.environ.get("MK_MODEL_DIR"))
    parser.add_argument("--ref", default=os.environ.get("MK_REF_DIR") or _SETUP.get("ref"))
    parser.add_argument("--baselines", default=os.environ.get("MK_BASELINES") or _SETUP.get("baselines"))
    parser.add_argument("--python", default=os.environ.get("MK_EVAL_PY") or _SETUP.get("python") or sys.executable)
    parser.add_argument("--scratch", default=os.environ.get("MK_EVAL_SCRATCH", "/dev/shm"))
    args = parser.parse_args()
    t_begin = time.time()
    workspace = str(Path(args.workspace).resolve())
    scratch = Path(args.scratch) / f"mk-eval-{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(args.ref) / "meta.json").read_text())
    global HBM_GB_S
    on = card(args.baselines, _SETUP)
    HBM_GB_S = on["hbm_gb_s"]
    result: dict = {"status": "error", "quick": args.quick, "build": {}, "gates": {}, "perf": {}, "trace": {},
                    "milestones_passed": [], "reference": {"hf_bf16_rel_l2": meta["hf_bf16_rel_l2"],
                                                           "hf_bf16_top1_agreement": meta["hf_bf16_top1_agreement"],
                                                           "tolerance_factor": TOLERANCE, "hbm_gb_s": HBM_GB_S,
                                                           "hw": on["hw"], "hbm_source": on["hbm_source"],
                                                           "dtype": meta.get("reference_dtype", "float64"),
                                                           "model_dir": meta.get("model_dir")}}

    def finish(message: str) -> int:
        result["seconds"] = round(time.time() - t_begin, 1)
        result["message"] = message
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
        subprocess.run(["rm", "-rf", str(scratch)])
        print("MK_EVAL " + json.dumps({k: result.get(k) for k in ("status", "headline", "milestones_passed", "seconds", "message")}))
        return 0

    def child(mode: str, extra: list[str], timeout: float, prefix: list[str] | None = None) -> tuple[dict, dict | None]:
        out = scratch / f"{mode}.json"
        cmd = (prefix or []) + [args.python, str(HERE / "_engine_runner.py"), "--workspace", workspace, "--model-dir", args.model_dir,
                                "--ref", args.ref, "--mode", mode, "--out", str(out)] + extra
        done = run(cmd, timeout * on["timeout_scale"], cwd=workspace)
        return done, (json.loads(out.read_text()) if done["rc"] == 0 and out.exists() else None)

    # ---- build
    build_script = Path(workspace) / "engine" / "build.sh"
    if not (Path(workspace) / "engine" / "api.py").exists():
        result["status"] = "build_failed"
        return finish("engine/api.py is missing")
    if build_script.exists():
        built = run(["bash", str(build_script)], 1500, cwd=workspace)
        result["build"] = {"rc": built["rc"], "seconds": built["seconds"], "tail": built["tail"][-1500:]}
        if built["rc"] != 0:
            result["status"] = "build_failed"
            return finish(f"engine/build.sh failed (rc {built['rc']})")
    else:
        result["build"] = {"rc": 0, "seconds": 0.0, "note": "no engine/build.sh"}
    result["milestones_passed"].append("build")

    # ---- check
    n_check = 2 if args.quick else 4
    done, report = child("check", ["--check-sequences", str(n_check)], 3600)
    if report is None:
        result["gates"]["run"] = {"pass": False, "rc": done["rc"], "tail": done["tail"][-2500:]}
        return finish(f"the engine failed during the correctness run (rc {done['rc']})")
    result["load_seconds"] = report["load_seconds"]
    ids = np.load(Path(args.ref) / "ids.npy")
    ref = np.load(Path(args.ref) / "logits64.npy")
    T, P = meta["check_tokens"], meta["prompt_tokens"]
    single = np.load(scratch / "check.single.npy")
    batched = np.load(scratch / "check.batched.npy")
    free = np.load(scratch / "check.free.npy")
    finite = bool(np.isfinite(single).all() and np.isfinite(batched).all())
    limit = {k: TOLERANCE * meta["hf_bf16_rel_l2"][k] for k in ("median", "max")}
    gates = result["gates"]
    gates["finite"] = {"pass": finite}
    if finite:
        err_single, err_batched = stats(rel_l2(single, ref[:n_check])), stats(rel_l2(batched, ref))
        gates["numeric"] = {"pass": all(e[k] <= limit[k] for e in (err_single, err_batched) for k in ("median", "max")),
                            "rel_l2_single": err_single, "rel_l2_batched": err_batched, "limit": limit}
        ref_top1 = ref.argmax(-1)
        agree = float(((single.argmax(-1) == ref_top1[:n_check]).mean() + (batched.argmax(-1) == ref_top1).mean()) / 2)
        gates["tokens"] = {"pass": agree >= meta["hf_bf16_top1_agreement"] - 0.01, "top1_agreement_with_float64": agree,
                           "hf_bf16": meta["hf_bf16_top1_agreement"]}
        diff = np.abs(single - batched[:n_check])
        gates["batch"] = {"pass": bool(stats(rel_l2(single, batched[:n_check]))["max"] <= limit["max"]),
                          "bitwise_equal": bool((diff == 0).all()), "max_abs_logit_diff": float(diff.max()),
                          "note": "same sequence alone and inside a batch of %d" % ref.shape[0]}
        http_prompts, http_expect = ids[:, :P].tolist(), free.tolist()
        target = ids[:, P:T]
        match = (free == target)
        prefix = np.where(match.all(1), match.shape[1], match.argmin(1))
        result["free_run_tokens"] = free.tolist()  # the engine's own step-by-step greedy tokens (the online evaluation compares with them)
        gates["free_run"] = {"pass": True, "informative_only": True, "mean_matching_prefix": float(prefix.mean() / match.shape[1]),
                             "sequences_identical": int(match.all(1).sum()), "of": int(match.shape[0]),
                             "note": "greedy continuation against the Hugging Face BF16 one; diverges after any near-tie"}
        if report.get("has_generate"):
            gen = np.load(scratch / "check.gen.npy")
            same = (gen == free) if gen.shape == free.shape else np.zeros_like(free, dtype=bool)
            gen_prefix = np.where(same.all(1), same.shape[1], same.argmin(1))
            gates["generate"] = {"pass": bool(gen.shape == free.shape and gen_prefix.mean() / same.shape[1] >= 0.75
                                              and same.all(1).sum() * 2 >= same.shape[0]),
                                 "mean_matching_prefix": float(gen_prefix.mean() / same.shape[1]),
                                 "sequences_identical": int(same.all(1).sum()), "of": int(same.shape[0]),
                                 "note": "the engine's own greedy loop against its step-by-step greedy; they differ only at exact ties"}
    if not finite:
        http_prompts = http_expect = None
    has_generate = bool(report.get("has_generate"))
    correct = finite and all(gates[g]["pass"] for g in ("numeric", "tokens", "batch"))
    long_positions = long_ref = None
    if (Path(args.ref) / "long_positions.npy").exists() and (scratch / "check.long.npy").exists():
        long_positions = np.load(Path(args.ref) / "long_positions.npy")
        long_ref = np.load(Path(args.ref) / "long_logits64.npy")
        long = np.load(scratch / "check.long.npy")
        long_limit = {k: TOLERANCE * meta["hf_bf16_rel_l2_long"][k] for k in ("median", "max")}
        if np.isfinite(long).all():
            err_long = stats(rel_l2(long, long_ref))
            agree_long = float((long.argmax(-1) == long_ref.argmax(-1)).mean())
            per = rel_l2(long, long_ref)                                    # [R, long positions]
            hf_path = Path(args.ref) / "hf_bf16_long_err.npy"
            hf_per = np.load(hf_path) if hf_path.exists() else None
            order = np.argsort(per.reshape(-1))[::-1][:5]
            worst = [{"sequence": int(i // per.shape[1]), "position": int(long_positions[i % per.shape[1]]), "rel_l2": float(per.reshape(-1)[i]),
                      "hf_bf16": None if hf_per is None else float(hf_per.reshape(-1)[i]),
                      "top1_same": bool(long.reshape(-1, long.shape[-1])[i].argmax() == long_ref.reshape(-1, long_ref.shape[-1])[i].argmax())} for i in order]
            gates["numeric_long"] = {"pass": all(err_long[k] <= long_limit[k] for k in ("median", "max"))
                                     and agree_long >= meta["hf_bf16_top1_agreement_long"] - 0.01,
                                     "rel_l2": err_long, "limit": long_limit, "top1_agreement_with_float64": agree_long,
                                     "hf_bf16_top1": meta["hf_bf16_top1_agreement_long"],
                                     "positions": int(per.size), "over_max_limit": int((per > long_limit["max"]).sum()), "worst": worst,
                                     "hf_bf16_p99": None if hf_per is None else float(np.sort(hf_per.reshape(-1))[int(0.99 * hf_per.size)]),
                                     "note": "batch of %d, positions %d to %d (every %d-th)" % (long.shape[0], long_positions[0], long_positions[-1], meta["long_every"])}
        else:
            gates["numeric_long"] = {"pass": False, "note": "non-finite logits at a long position"}
        correct = correct and gates["numeric_long"]["pass"]
        del long
    if correct:
        result["milestones_passed"].append("correct")
    del single, batched

    # ---- time
    segments = 1 if args.quick else 3
    done, report = child("time", ["--segments", str(segments)], 1800)
    if report is None:
        result["perf"]["run"] = {"rc": done["rc"], "tail": done["tail"][-2500:]}
        result["status"] = "incorrect" if not correct else "error"
        return finish(f"the engine failed during the timed run (rc {done['rc']})")
    result["engine_stats"] = report.get("engine_stats")
    result["gpu_memory_used_gb"] = report.get("gpu_memory_used_gb")
    result["gpu_memory_used_gb_per_card"] = report.get("gpu_memory_used_gb_per_card")
    verified = True
    for batch, timing in report["timing"].items():
        B = int(batch)
        walls = sorted(timing["walls_s"])
        step_s = walls[len(walls) // 2] / timing["steps"]
        context = (timing["context_start"] + timing["context_end"]) / 2
        need = decode_bytes(args.model_dir, B, context)
        probe_err = []
        long_index = {int(t): i for i, t in enumerate(long_positions)} if long_positions is not None else {}
        for segment in timing.get("probe_segments", []):
            for t in timing["probe_steps"]:
                probe = np.load(scratch / f"time.probe.b{B}.s{segment}.t{t}.npy")
                target = ref[:B, t] if t < T else long_ref[:B, long_index[t]]
                probe_err.append(float(rel_l2(probe, target).max()) if np.isfinite(probe).all() else float("inf"))
        probe_limit = max(limit["max"], TOLERANCE * meta.get("hf_bf16_rel_l2_long", {}).get("max", 0.0))
        ok = timing["last_logits_finite"] and bool(probe_err) and max(probe_err) <= probe_limit
        verified = verified and ok
        result["perf"][f"b{B}"] = {"step_us": step_s * 1e6, "step_us_min": walls[0] / timing["steps"] * 1e6,
                                   "step_us_max": walls[-1] / timing["steps"] * 1e6, "tok_s": B / step_s,
                                   "mbu": need["total"] / step_s / (HBM_GB_S * 1e9),
                                   "mbu_high": need["total_high"] / step_s / (HBM_GB_S * 1e9), "bytes_per_step": need,
                                   "segments": len(walls), "steps": timing["steps"], "mean_context": context,
                                   "timed_steps_verified": ok, "probe_rel_l2_max": max(probe_err) if probe_err else None,
                                   "probes_checked": len(probe_err), "probe_steps": timing["probe_steps"]}
        if timing.get("w1_walls_s"):
            w1_step = sorted(timing["w1_walls_s"])[len(timing["w1_walls_s"]) // 2] / timing["w1_steps"]
            w1_need = decode_bytes(args.model_dir, B, 64 + timing["w1_steps"] / 2)
            result["perf"][f"b{B}"].update(w1_like_step_us=w1_step * 1e6, w1_like_tok_s=B / w1_step,
                                           w1_like_mbu=w1_need["total"] / w1_step / (HBM_GB_S * 1e9), w1_like_steps=timing["w1_steps"])
        if timing.get("w1_generate_walls_s"):
            g = sorted(timing["w1_generate_walls_s"])[len(timing["w1_generate_walls_s"]) // 2] / timing["w1_steps"]
            result["perf"][f"b{B}"].update(w1_like_generate_step_us=g * 1e6, w1_like_generate_tok_s=B / g)
        if timing.get("generate_walls_s"):
            gen_walls = sorted(timing["generate_walls_s"])
            gen_step = gen_walls[len(gen_walls) // 2] / timing["steps"]
            result["perf"][f"b{B}"].update(generate_step_us=gen_step * 1e6, generate_tok_s=B / gen_step,
                                           generate_mbu=need["total"] / gen_step / (HBM_GB_S * 1e9),
                                           generate_shape_ok=timing.get("generate_shape_ok"))
    result["perf"]["timing"] = ("step_us: step, no sampling, host-driven; wall clock around the timed steps with one final synchronize. "
                                "w1_like_step_us: prompt 64, then 1024 greedy tokens with the token read on the host every step "
                                "(the baselines' workload without HTTP and detokenization)")
    gates["timed_steps"] = {"pass": verified, "note": "logits kept during the timed run, checked against float64 afterwards"}
    correct = correct and verified

    # ---- trace
    if not (args.quick or args.no_trace):
        trace_steps = 128
        trace_base = scratch / "trace"
        prefix = ["nsys", "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--sample=none", "--cpuctxsw=none",
                  "--force-overwrite=true", "--output", str(trace_base)]
        done, report = child("trace", ["--timed-steps", str(trace_steps), "--warm-steps", "32"], 1500, prefix=prefix)
        if report is None:
            result["trace"] = {"error": f"traced run failed (rc {done['rc']})", "tail": done["tail"][-1200:]}
        else:
            exported = run(["nsys", "export", "--type=sqlite", "--force-overwrite=true", "--output", str(trace_base) + ".sqlite",
                            str(trace_base) + ".nsys-rep"], 900)
            try:
                result["trace"] = analyse_trace(Path(str(trace_base) + ".sqlite"), trace_steps) if exported["rc"] == 0 \
                    else {"error": "nsys export failed", "tail": exported["tail"][-800:]}
                if exported["rc"] == 0 and has_generate:
                    result["trace_generate"] = analyse_trace(Path(str(trace_base) + ".sqlite"), trace_steps, "mk_generate")
            except Exception as error:
                result["trace"] = {"error": f"{type(error).__name__}: {error}"}

    # ---- the headline's stage: the engine behind an HTTP server, measured by the baselines' client
    use_http = not (args.no_http or args.quick)
    http = None
    if use_http and correct:
        tools_py = args.tools_python or args.python
        http = http_stage.measure(workspace, args.model_dir, args.python, tools_py, scratch / "http",
                                  os.environ.get("MK_SERVER_CPUS"), os.environ.get("MK_CLIENT_CPUS"),
                                  startup_s=1800 * on["timeout_scale"], run_s=2400 * on["timeout_scale"], read_s=120 * on["timeout_scale"],
                                  prompts=http_prompts, expect=http_expect)
        result["http"] = http
        if http.get("tokens") is not None:
            gates["http_tokens"] = http["tokens"]
            correct = correct and bool(http["tokens"].get("pass"))
        if correct and not http.get("ok"):
            result["status"] = "error"
            return finish("the engine passed the gates but could not be measured over HTTP: " + str(http.get("error")))
        for b in ("b1", "b8"):
            if b in result["perf"] and "step_us" in (http.get(b) or {}):
                result["perf"][b].update(http_step_us=http[b]["step_us"], http_tok_s=http[b]["decode_tok_s"]["median"])
    result["perf"]["diagnostics"] = ("step_us, generate_step_us, w1_like_* and mbu are timed inside the engine's process and are diagnostics; "
                                     "http_step_us is the headline's calibre" if use_http else "no HTTP stage in this evaluation")

    # ---- verdict
    result["status"] = "valid" if correct else "incorrect"
    passed = result["milestones_passed"]
    if correct:
        b1, b8 = result["perf"].get("b1"), result["perf"].get("b8")
        # In process: microseconds per token at B=1 through the engine's best valid path: host-driven steps, or the
        # engine's own loop once it exists and reproduces the step-by-step tokens. The headline without HTTP stage.
        best, path = b1["step_us"], "step"
        if gates.get("generate", {}).get("pass") and b1.get("generate_step_us") and b1["generate_step_us"] < best:
            best, path = b1["generate_step_us"], "generate"
        result["in_process"] = {"metric": "step_us_b1", "value": best, "path": path}
        if use_http:
            result["headline"] = {"metric": "http_step_us_b1", "value": http["b1"]["step_us"], "lower_is_better": True,
                                  "path": http.get("contract"), "b8": http["b8"]["step_us"]}
        else:
            result["headline"] = {"metric": "step_us_b1", "value": best, "lower_is_better": True, "path": path}
        if gates.get("generate", {}).get("pass"):
            passed.append("device_loop")
        trace = result.get("trace_generate") if gates.get("generate", {}).get("pass") and "error" not in (result.get("trace_generate") or {"error": 1}) else result["trace"]
        if trace and "error" not in trace:
            if trace["kernel_launches_per_step"] <= 1.5:
                passed.append("single_launch")
            if trace["host_syncs_per_step"] <= 0.05 and trace["gaps_ge_20us_per_step"] <= 0.05:
                passed.append("gpu_never_waits")  # was "no_host_in_loop": the host may still launch every step
        if args.baselines and Path(args.baselines).exists():
            lines = json.loads(Path(args.baselines).read_text())
            result["baselines"] = lines
            def in_process(b: str) -> float:
                return min(result["perf"][b]["step_us"], result["perf"][b].get("generate_step_us") or 1e30
                           if gates.get("generate", {}).get("pass") else 1e30)

            def figure(b: str) -> float:  # what is compared with the baselines: the HTTP figure when there is one
                return result["perf"][b]["http_step_us"] if use_http else in_process(b)

            speed = {name: {b: entry[b] / figure(b) for b in ("b1", "b8") if b in entry and b in result["perf"]}
                     for name, entry in lines.get("systems", {}).items()}
            result["speedup"] = speed
            result["speedup_calibre"] = "http_step_us against the baselines' HTTP numbers" if use_http else \
                "in-process step time against the baselines' HTTP numbers: an upper bound"
            if use_http:
                result["speedup_in_process_diagnostic"] = {name: {b: entry[b] / in_process(b) for b in ("b1", "b8") if b in entry and b in result["perf"]}
                                                           for name, entry in lines.get("systems", {}).items()}
            engines = [s for name, s in speed.items() if lines["systems"][name].get("general_engine")]
            # Other megakernels for the same model (MPK for Qwen3-8B, Cohere's hand-written one for North Mini Code).
            for name, entry in lines.get("systems", {}).items():
                if not entry.get("general_engine") and len(speed.get(name, {})) == 2 and all(v >= 1.0 for v in speed[name].values()):
                    passed.append(f"beats_{name}")
            if engines and all(min(s[b] for s in engines) >= 1.0 for b in ("b1", "b8")):
                passed.append("parity")
            if engines and all(min(s[b] for s in engines) >= 1.10 for b in ("b1", "b8")):
                passed.append("h0_step")
            # The same comparison on the workload the baselines were measured on (sampling and a host read included).
            def w1_best(b: str) -> float | None:
                # host-driven greedy, or the engine's own loop once it reproduces the step-by-step tokens
                perf = result["perf"].get(b, {})
                values = [perf.get("w1_like_step_us")] + ([perf.get("w1_like_generate_step_us")] if gates.get("generate", {}).get("pass") else [])
                values = [v for v in values if v]
                return min(values) if values else None

            w1 = {name: {b: lines["systems"][name][b] / w1_best(b) for b in ("b1", "b8") if b in lines["systems"][name] and w1_best(b)}
                  for name in lines.get("systems", {})}
            result["speedup_w1_like"] = w1
            w1_engines = [s for name, s in w1.items() if lines["systems"][name].get("general_engine") and len(s) == 2]
            if not use_http and w1_engines and all(min(s[b] for s in w1_engines) >= 1.10 for b in ("b1", "b8")):
                passed.append("h0_w1_like")
        if use_http:
            return finish(f"valid: {http['b1']['step_us']:.0f} us/step at B=1 and {http['b8']['step_us']:.0f} at B=8 over HTTP "
                          f"({http.get('contract')}); in process {best:.0f} us/step at B=1 ({path}, MBU {b1['mbu']:.3f})")
        return finish(f"valid: {b1['step_us']:.0f} us/step at B=1 (MBU {b1['mbu']:.3f})"
                      + (f", {b8['step_us']:.0f} us/step at B=8" if b8 else ""))
    failed = [g for g, v in gates.items() if not v.get("pass")]
    return finish("incorrect: failed " + ", ".join(failed))


if __name__ == "__main__":
    raise SystemExit(main())
