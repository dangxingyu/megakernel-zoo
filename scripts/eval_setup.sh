#!/usr/bin/env bash
# Run once per pod by the loop's worker (setup_run --eval-setup "bash scripts/eval_setup.sh"), in the harness:
# the evaluator's Python environment and the float64 reference of the run's model. Model and card come from the
# environment: MK_MODEL (Hugging Face id, default Qwen/Qwen3-8B) and MK_HW (h100 | b200; default: the card the pod
# has), e.g. --eval-setup "MK_MODEL=RecursiveSI/flint MK_HW=b200 bash scripts/eval_setup.sh". The reference lines are
# eval/baselines_<model>_<card>.json (<model>: the id's last part in lower case). Writes $MK_ROOT/eval_env.json:
#   "env":     added to the environment of every job in the pod (the agent's `mk exec` included): a Python with torch
#   "private": read by eval/kernel_eval.py only (where the reference and the reference lines are)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MK_REPO"
mk_tools
MK_MODEL="${MK_MODEL:-Qwen/Qwen3-8B}"
py="$MK_ROOT/venvs/eval/bin/python"
if [ ! -x "$py" ]; then
  uv venv -q "$MK_ROOT/venvs/eval" --python python3.12
  extra_pip=(); case "$(echo "$MK_MODEL" | tr 'A-Z' 'a-z')" in *gpt-oss*) extra_pip=(triton "kernels>=0.18.2,<0.19") ;; esac   # MXFP4 experts in their own format
  uv pip install -q --python "$py" torch transformers accelerate numpy tokenizers safetensors nvidia-ml-py pytest "${extra_pip[@]}" --torch-backend=auto
fi
# the evaluation's HTTP stage runs the workload client and the front end under the tools Python (aiohttp, tokenizers, NVML)
uv pip install -q --python "$MK_TOOLS_PY" nvidia-ml-py
"$py" -c 'import torch, transformers; assert torch.cuda.is_available(); print(f"MK_ENV eval torch={torch.__version__} cuda={torch.version.cuda} transformers={transformers.__version__}")'
name="$(echo "$MK_MODEL" | tr '/A-Z' '-a-z')"
ref="$MK_ROOT/ref/$name"
ref_version="${MK_REF_VERSION:-v3}"   # v3: v2 (float64 logits along 1152 positions) plus the BF16 path's KL; v1 stopped at 128
[ -f "$ref/meta.json" ] || store get "$MK_S3/ref/$name-$ref_version" "$ref" > "$MK_ROOT/logs/ref-get.log" 2>&1
[ -f "$ref/meta.json" ] || { echo "no reference for $MK_MODEL in the store: run scripts/eval_selftest.sh once"; exit 3; }
hidden="$MK_ROOT/ref/$name-hidden"   # suite v2: kept out of the run's workspace; only kernel_eval.py reads it
[ -f "$hidden/meta.json" ] || store get "$MK_S3/ref/$name-hidden-${MK_HIDDEN_VERSION:-v1}" "$hidden" > "$MK_ROOT/logs/ref-hidden-get.log" 2>&1
[ -f "$hidden/meta.json" ] || { echo "MK_WARN no hidden reference for $MK_MODEL: the hidden gate is skipped"; hidden=""; }
short="$(echo "${MK_MODEL##*/}" | tr 'A-Z' 'a-z')"
hw="$(mk_hw)"
baselines="$MK_REPO/eval/baselines_${short}_${hw}.json"
[ -f "$baselines" ] || echo "MK_WARN no $baselines: the result will have no speedup block"
# The production trace of the online evaluation (Mooncake FAST'25, tool & agent), pinned to a commit.
trace="$MK_ROOT/ref/toolagent_trace.jsonl"
[ -s "$trace" ] || curl -fsSL -o "$trace" "https://raw.githubusercontent.com/kvcache-ai/Mooncake/aaa49180771f/FAST25-release/traces/toolagent_trace.jsonl"
[ "$(sha256sum "$trace" | cut -c1-16)" = 48a2db1a13d3bc05 ] || { echo "trace checksum mismatch"; exit 4; }
cat > "$MK_ROOT/eval_env.json" <<JSON
{"env": {"MK_PY": "$py", "PATH_PREPEND": "$MK_ROOT/venvs/eval/bin", "MK_TRACE": "$trace", "MK_HW": "$hw", "MK_MODEL": "$MK_MODEL"},
 "private": {"python": "$py", "tools_python": "$MK_TOOLS_PY", "ref": "$ref", "ref_hidden": "$hidden", "baselines": "$baselines", "trace": "$trace", "hw": "$hw",
             "online_baselines": "$MK_REPO/eval/baselines_online_${short}_${hw}.json"}}
JSON
echo "MK_ENV eval_env $(tr -d '\n' < "$MK_ROOT/eval_env.json")"
