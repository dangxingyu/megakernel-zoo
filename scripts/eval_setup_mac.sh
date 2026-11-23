#!/usr/bin/env bash
# The Apple-silicon counterpart of scripts/eval_setup.sh: run once on the Mac before a run's first evaluation. No
# nvidia-smi, no CUDA, no object store: the evaluator's Python is the repository's .venv-mac (MLX, mlx-lm, NumPy,
# aiohttp, tokenizers), the model is a local Hugging Face snapshot and the float64 reference a local directory.
#
#   bash scripts/eval_setup_mac.sh
#   MK_MODEL_DIR=~/models/Inkling-2-Mini MK_REF_DIR=~/mk-ref/thinkingmachines-inkling-2-mini bash scripts/eval_setup_mac.sh
#
# Environment (all optional):
#   MK_MODEL      Hugging Face id                       default thinkingmachines/Inkling-2-Mini
#   MK_MODEL_DIR  the local snapshot                    default ~/models/<the id's last part>
#   MK_REF_DIR    the float64 reference (ids.npy, meta.json, logits64.npy, long_positions.npy, long_logits64.npy;
#                 eval/make_reference.py on a GPU machine)  default ~/mk-ref/<org>-<model> in lower case
#   MK_REF_HIDDEN a hidden reference (suite v2)          default none: the hidden gate is skipped
#   MK_HW         the card tag (eval/modelinfo.py CARDS) default from the chip's name: "Apple M5 Max" -> m5max
#   MK_TRACK      a | b | c                              default c (exact speculative decoding)
#   MK_PY         the evaluator's Python                 default <repo>/.venv-mac/bin/python
#   MK_ROOT       where eval_env.json goes               default /tmp/mk (as on the pods; gone after a reboot: run again)
# Writes $MK_ROOT/eval_env.json ("env": for the agent's jobs, "private": read by eval/kernel_eval.py only) with
# track c, card m5max and the reference lines eval/baselines_<model>_<card>[_c].json.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MK_ROOT="${MK_ROOT:-/tmp/mk}"
mkdir -p "$MK_ROOT/logs" "$MK_ROOT/ref"
[ "$(uname -s)" = Darwin ] || { echo "eval_setup_mac.sh is for macOS; on a GPU pod use scripts/eval_setup.sh"; exit 2; }

MK_MODEL="${MK_MODEL:-thinkingmachines/Inkling-2-Mini}"
py="${MK_PY:-$repo/.venv-mac/bin/python}"
[ -x "$py" ] || { echo "no $py: make the Mac venv first (uv venv .venv-mac --python 3.12, then mlx, mlx-lm with the model's support, numpy, aiohttp, tokenizers)"; exit 2; }
"$py" - <<'PY'
import mlx.core as mx, numpy, aiohttp, tokenizers  # noqa: F401
import mlx_lm
assert mx.metal.is_available(), "MLX sees no Metal GPU"
info = mx.device_info()
print(f"MK_ENV eval mlx={mx.__version__} mlx_lm={getattr(mlx_lm, '__version__', '?')} numpy={numpy.__version__} "
      f"device={info['device_name'].replace(' ', '_')} memory_gb={info['memory_size'] / 1e9:.1f} "
      f"working_set_gb={info['max_recommended_working_set_size'] / 1e9:.1f}")
PY

short="$(echo "${MK_MODEL##*/}" | tr 'A-Z' 'a-z')"
name="$(echo "$MK_MODEL" | tr '/A-Z' '-a-z')"
model_dir="${MK_MODEL_DIR:-$HOME/models/${MK_MODEL##*/}}"
[ -f "$model_dir/config.json" ] || { echo "no checkpoint at $model_dir (set MK_MODEL_DIR)"; exit 3; }
# Every safetensors file complete: its header's last offset + header length + 8 == the file's size (a download in
# progress, e.g. aria2 with a .aria2 control file next to it, fails this).
"$py" - "$model_dir" <<'PY' || exit 3
import json, struct, sys
from pathlib import Path
root, bad = Path(sys.argv[1]), []
for f in sorted(root.glob("*.safetensors")):
    with open(f, "rb") as h:
        n = struct.unpack("<Q", h.read(8))[0]
        header = json.loads(h.read(n))
    end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
    ok = 8 + n + end == f.stat().st_size and not Path(str(f) + ".aria2").exists()
    if not ok:
        bad.append(f"{f.name}: {f.stat().st_size} of {8 + n + end} bytes")
    print(f"MK_ENV weights {f.name} {f.stat().st_size / 1e9:.2f} GB {'complete' if ok else 'INCOMPLETE'}")
if bad:
    print("MK_ERROR incomplete checkpoint: " + "; ".join(bad))
    sys.exit(3)
PY

ref="${MK_REF_DIR:-$HOME/mk-ref/$name}"
missing=""
for f in ids.npy meta.json logits64.npy long_positions.npy long_logits64.npy; do [ -f "$ref/$f" ] || missing="$missing $f"; done
if [ -n "$missing" ]; then
  echo "MK_WARN the float64 reference at $ref lacks:$missing. The evaluator stops with status error until it is there:"
  echo "MK_WARN copy the reference made on a GPU machine (eval/make_reference.py; ~1.6 GB) to $ref, or set MK_REF_DIR"
fi
hidden="${MK_REF_HIDDEN:-}"
[ -z "$hidden" ] || [ -f "$hidden/meta.json" ] || { echo "MK_WARN no hidden reference at $hidden: the hidden gate is skipped"; hidden=""; }

if [ -n "${MK_HW:-}" ]; then
  hw="$(echo "$MK_HW" | tr 'A-Z' 'a-z')"
else
  hw="$("$py" -c "import sys; sys.path.insert(0, '$repo/eval'); import modelinfo; print(modelinfo.hw_of_gpu_name('$(sysctl -n machdep.cpu.brand_string)') or '')")"
  [ -n "$hw" ] || { echo "this chip ($(sysctl -n machdep.cpu.brand_string)) has no entry in eval/modelinfo.py CARDS: add one (scripts/mac_membw.py measures its bandwidth) or set MK_HW"; exit 4; }
fi
track="$(echo "${MK_TRACK:-c}" | tr 'A-Z' 'a-z')"
baselines="$repo/eval/baselines_${short}_${hw}.json"
[ "$track" = c ] && baselines="$repo/eval/baselines_${short}_${hw}_c.json"
[ -f "$baselines" ] || echo "MK_WARN no $baselines: the result will have no speedup block"
# The production trace of the online evaluation (Mooncake FAST'25, tool & agent), pinned to a commit; optional here.
trace="$MK_ROOT/ref/toolagent_trace.jsonl"
[ -s "$trace" ] || curl -fsSL -o "$trace" "https://raw.githubusercontent.com/kvcache-ai/Mooncake/aaa49180771f/FAST25-release/traces/toolagent_trace.jsonl" \
  || { echo "MK_WARN could not fetch the online trace (only eval/online_eval.py needs it)"; rm -f "$trace"; }
if [ -s "$trace" ] && [ "$(shasum -a 256 "$trace" | cut -c1-16)" != 48a2db1a13d3bc05 ]; then echo "trace checksum mismatch"; exit 4; fi
cat > "$MK_ROOT/eval_env.json" <<JSON
{"env": {"MK_PY": "$py", "PATH_PREPEND": "$(dirname "$py")", "MK_TRACE": "$trace", "MK_HW": "$hw", "MK_MODEL": "$MK_MODEL",
         "MK_MODEL_DIR": "$model_dir", "MK_DEVICE": "metal", "MK_EVAL_BATCHES": "1"},
 "private": {"python": "$py", "tools_python": "$py", "ref": "$ref", "ref_hidden": "$hidden", "baselines": "$baselines", "trace": "$trace", "hw": "$hw", "track": "$track",
             "model_dir": "$model_dir", "online_baselines": "$repo/eval/baselines_online_${short}_${hw}.json"}}
JSON
echo "MK_ENV eval_env $(tr -d '\n' < "$MK_ROOT/eval_env.json")"
