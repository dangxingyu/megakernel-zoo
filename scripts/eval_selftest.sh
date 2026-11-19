#!/usr/bin/env bash
# bash scripts/eval_selftest.sh [MODEL_REPO]: make (or fetch) the float64 reference for the model and run the
# kernel evaluator on the starter engine (Hugging Face behind the contract). The starter must come out "valid":
# if it does not, the evaluator or its tolerance is wrong, not the engine.
# The card is MK_HW (h100 | b200; default: the machine's card); the reference lines are eval/baselines_<model>_<card>.json.
# MK_REF_DTYPE (auto | float64 | float32) and MK_REF_CARDS (spread the reference over N cards) go to eval/make_reference.py; MK_REF_REMAKE=1 makes the reference
# again although the store has one; MK_SELFTEST_MODES ("quick full" by default; "none": only the reference) picks
# the evaluations. A reference is only token ids and logits: it can be made on another card than the run's.
# MK_SELFTEST_WS (a directory with engine/, default starter) is the engine evaluated; MK_SELFTEST_ARGS are passed on to
# eval/kernel_eval.py; MK_HTTP_CHECK="TOKENS REPEATS" runs only eval/http_stage.py on that engine (a quick look at the shim).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MK_REPO"
env_report
mk_tools || exit 10
MK_MODEL="${1:-Qwen/Qwen3-8B}"
model="$(mk_model "$MK_MODEL")" || exit 11
source scripts/cuda_toolkit.sh > "$MK_ROOT/logs/cuda-toolkit.log" 2>&1; grep MK_ENV "$MK_ROOT/logs/cuda-toolkit.log"
set +e +u; set -o pipefail
t=$(date +%s)
py="$MK_ROOT/venvs/eval/bin/python"
extra_pip=(); case "$(echo "$MK_MODEL" | tr 'A-Z' 'a-z')" in *gpt-oss*) extra_pip=(triton "kernels>=0.18.2,<0.19") ;; esac   # MXFP4 experts in their own format (else transformers dequantizes them to BF16)
uv venv -q "$MK_ROOT/venvs/eval" --python python3.12 && uv pip install -q --python "$py" torch transformers accelerate numpy tokenizers safetensors nvidia-ml-py "${extra_pip[@]}" --torch-backend=auto
"$py" -c 'import torch, transformers; print(f"MK_ENV eval torch={torch.__version__} cuda={torch.version.cuda} transformers={transformers.__version__} gpu={torch.cuda.get_device_name(0)}")' || exit 12
stage "eval venv" ok $t

name="$(echo "$MK_MODEL" | tr '/A-Z' '-a-z')"
short="$(echo "${MK_MODEL##*/}" | tr 'A-Z' 'a-z')"
hw="$(mk_hw)"; export MK_HW="$hw"
ref="$MK_ROOT/ref/$name" key="$MK_S3/ref/$name-${MK_REF_VERSION:-v2}"
echo "MK_ENV model=$MK_MODEL short=$short hw=$hw model_dir=$model"
"$py" eval/modelinfo.py "$model" --hw "$hw" | cut -c1-3000
t=$(date +%s)
if [ -z "${MK_REF_REMAKE:-}" ] && store has "$key" > /dev/null 2>&1; then
  store get "$key" "$ref" > "$MK_ROOT/logs/ref-get.log" 2>&1 && stage "reference from store" ok $t
else
  ids=(); [ -f "eval/refs/$name/ids.npy" ] && [ -z "${MK_REF_OWN_SEQUENCES:-}" ] && ids=(--ids "eval/refs/$name/ids.npy")   # the sequences of our own runs
  "$py" eval/make_reference.py --model-dir "$model" --out "$ref" --dtype "${MK_REF_DTYPE:-auto}" --cards "${MK_REF_CARDS:-1}" "${ids[@]}" 2>&1 | grep -E "MK_REF|Error|error|Traceback" | cut -c1-1200
  [ -f "$ref/meta.json" ] || { stage "reference" failed $t; exit 13; }
  store put "$ref" "$key" "$MK_S3" 2>&1 | tail -n 1 | cut -c1-200
  stage "reference made" ok $t
fi
cat "$ref/meta.json" | tr -d '\n' | cut -c1-1500; echo
if [ -n "${MK_GATE_STUDY:-}" ]; then   # where the BF16 path's distance from the reference comes from (eval/gate_study.py)
  t=$(date +%s)
  "$py" eval/gate_study.py --model-dir "$model" --ref "$ref" 2>&1 | grep -E "MK_GATE|Error|error|Traceback|File " | cut -c1-4000
  stage "gate study" done $t
fi

uv pip install -q --python "$MK_TOOLS_PY" nvidia-ml-py
cpu_args=()
if [ "${MK_TARGET:-card}" = node ]; then  # a whole-node engine: most of the node's physical cores for its server
  phys=$(lscpu -p=core,socket 2>/dev/null | grep -v '^#' | sort -u | wc -l); cpu_args=(--server $(( phys - 12 )) --client 4)
fi
picked="$(python3 scripts/pick_cpus.py "${cpu_args[@]}")" && eval "$picked" && export MK_SERVER_CPUS="$SERVER_CPUS" MK_CLIENT_CPUS="$CLIENT_CPUS"
echo "MK_ENV ${picked:-no quiet cores picked}" | cut -c1-300
export MK_MODEL_DIR="$model" MK_REF_DIR="$ref" MK_EVAL_PY="$py" MK_BASELINES="$MK_REPO/eval/baselines_${short}_${hw}.json"
[ -f "$MK_BASELINES" ] || echo "MK_WARN no $MK_BASELINES: no speedup block"
export PATH="/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
work="$MK_ROOT/run/selftest-ws"; rm -rf "$work"
if [ -n "${MK_SELFTEST_WS:-}" ] && [ -f "$MK_SELFTEST_WS" ]; then mkdir -p "$work" && tar xzf "$MK_SELFTEST_WS" -C "$work"; else cp -r "${MK_SELFTEST_WS:-starter}" "$work"; fi
if [ -n "${MK_HTTP_CHECK:-}" ]; then
  t=$(date +%s); read -r tokens repeats <<< "$MK_HTTP_CHECK"
  (cd "$work" && { [ ! -f engine/build.sh ] || PATH="$MK_ROOT/venvs/eval/bin:$PATH" bash engine/build.sh > "$MK_ROOT/logs/build-http-check.log" 2>&1; })
  "$py" eval/http_stage.py --workspace "$work" --model-dir "$model" --out "$MK_ROOT/run/http-check.json" --max-tokens "$tokens" --repeats "${repeats:-1}" 2>&1 | grep -E "MK_HTTP|rror|Traceback" | cut -c1-3500
  stage "http check" done $t
fi
for kind in ${MK_SELFTEST_MODES:-quick full}; do
  [ "$kind" = none ] && continue   # reference only (a model whose reference needs a larger card than its run)
  mode=""; [ "$kind" = quick ] && mode="--quick"
  t=$(date +%s)
  "$py" eval/kernel_eval.py --workspace "$work" --out "$MK_ROOT/run/selftest${mode:+-quick}.json" $mode ${MK_SELFTEST_ARGS:-} 2>&1 | tail -n 3 | cut -c1-1500
  "$py" - "$MK_ROOT/run/selftest${mode:+-quick}.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
slim = {k: r.get(k) for k in ("status", "headline", "milestones_passed", "seconds", "load_seconds", "gpu_memory_used_gb", "speedup", "message", "reference")}
print("MK_SELFTEST " + json.dumps(slim))
print("MK_SELFTEST gates " + json.dumps({k: {a: b for a, b in v.items() if a != "note"} for k, v in r.get("gates", {}).items()}))
print("MK_SELFTEST perf " + json.dumps({k: ({a: b for a, b in v.items() if a != "bytes_per_step"} if isinstance(v, dict) else v) for k, v in r.get("perf", {}).items()}))
print("MK_SELFTEST bytes " + json.dumps({k: v.get("bytes_per_step") for k, v in r.get("perf", {}).items() if isinstance(v, dict)}))
print("MK_SELFTEST trace " + json.dumps(r.get("trace")))
print("MK_SELFTEST http " + json.dumps({k: v for k, v in (r.get("http") or {}).items() if not k.endswith("_log")})[:3000])
if (r.get("http") or {}).get("error"): print("MK_SELFTEST http_logs " + json.dumps({k: v for k, v in r["http"].items() if k.endswith("_log")})[:3000])
if r.get("build", {}).get("rc"): print("MK_SELFTEST build " + json.dumps(r["build"]))
for stage in ("gates", "perf"):
    if "run" in r.get(stage, {}): print(f"MK_SELFTEST {stage} run " + json.dumps(r[stage]["run"])[:3000])
PY
  stage "kernel_eval ${mode:-full}" done $t
done
store put "$MK_ROOT/run" "$MK_S3/runs/${MK_JOB}" "$MK_S3" 2>&1 | tail -n 1 | cut -c1-200
stage "eval selftest" ok
