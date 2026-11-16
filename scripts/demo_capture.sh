#!/usr/bin/env bash
# bash scripts/demo_capture.sh: record real answers for the demo. On ONE card each system of $DEMO_SYSTEMS in turn
# (mkserve:ENGINE = our engine behind `python -m mkserve serve`; sglang; vllm) is started and asked the same three
# questions over the streaming chat endpoint (scripts/demo_capture.py). The recordings (text and arrival time of every
# piece) are stored as $MK_S3/demo/$MK_JOB/demo.json. MK_MODEL, MKSERVE_CARD, COMPARE_IMAGES / COMPARE_SG_FLAGS /
# COMPARE_SG_ENV / COMPARE_VL_FLAGS / COMPARE_SG_BASE as in scripts/compare.sh.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MK_REPO"
mkdir -p "$MK_ROOT/demo"
env_report
mk_tools || exit 10
model="$(mk_model "${MK_MODEL:-Qwen/Qwen3-8B}")" || exit 11
bash scripts/cuda_toolkit.sh > "$MK_ROOT/logs/cuda-toolkit.log" 2>&1 || { tail -n 20 "$MK_ROOT/logs/cuda-toolkit.log" | cut -c1-300; exit 16; }
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
set +e +u; set -o pipefail
read -r -a systems <<< "${DEMO_SYSTEMS:-mkserve:online-b sglang vllm}"
read -r -a sg_base <<< "${COMPARE_SG_BASE---enable-torch-compile}"; read -r -a sg_more <<< "${COMPARE_SG_FLAGS:-}"; read -r -a vl_flags <<< "${COMPARE_VL_FLAGS:-}"
sg_flags=("${sg_base[@]}" "${sg_more[@]}"); images="${COMPARE_IMAGES:-0}"; spy="$MK_ROOT/venvs/serve/bin/python"
if [ "$images" = 1 ]; then
  source scripts/ink/img.sh; sg_flags=("${sg_more[@]}")
  SGL_IMAGE="${COMPARE_SGL_IMAGE:-lmsysorg/sglang@sha256:198af079c599cbf37ad371faca16c988a71cea771b3c51fbe21d018f6e400c9e}"
fi
for system in "${systems[@]}"; do
  t=$(date +%s)
  case "$system" in
    sglang) if [ "$images" = 1 ]; then img_pull sglang "$SGL_IMAGE" > "$MK_ROOT/logs/install-sglang.log" 2>&1 || { tail -n 20 "$MK_ROOT/logs/install-sglang.log"; exit 12; }
            else SGLANG_VERSION="${SGLANG_VERSION:-0.5.18}" bash baselines/sglang/install.sh > "$MK_ROOT/logs/install-sglang.log" 2>&1 || { tail -n 30 "$MK_ROOT/logs/install-sglang.log" | cut -c1-300; exit 12; }; fi ;;
    vllm)   bash baselines/vllm/install.sh > "$MK_ROOT/logs/install-vllm.log" 2>&1 || { tail -n 30 "$MK_ROOT/logs/install-vllm.log" | cut -c1-300; exit 12; } ;;
    mkserve:*) [ -x "$spy" ] || { uv venv -q "$MK_ROOT/venvs/serve" --python python3.12 && uv pip install -q --python "$spy" torch numpy safetensors tokenizers aiohttp jinja2 huggingface_hub --torch-backend=auto; } \
                 > "$MK_ROOT/logs/serve-venv.log" 2>&1 || { tail -n 20 "$MK_ROOT/logs/serve-venv.log" | cut -c1-300; exit 12; } ;;
  esac
  stage "prepare $system" ok $t
done
picked="$(python3 scripts/pick_cpus.py 2>&1)" && eval "$picked" || { SERVER_CPUS="0-$(( $(nproc) - 5 ))"; CLIENT_CPUS="$(( $(nproc) - 4 ))-$(( $(nproc) - 1 ))"; }
for system in "${systems[@]}"; do
  tag="${system//:/-}"; t=$(date +%s); served=m
  # a port of its own for every server: the previous one's socket can outlive its process for a moment
  export MK_PORT="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
  case "$system" in
    sglang) if [ "$images" = 1 ]; then SGL_IMAGE_ENV="${COMPARE_SG_ENV:-}" taskset -c "$SERVER_CPUS" bash baselines/sglang/serve_image.sh "$model" "${sg_flags[@]}" > "$MK_ROOT/logs/server-$tag.log" 2>&1 & server=$!
            else taskset -c "$SERVER_CPUS" bash baselines/sglang/serve.sh "$model" "${sg_flags[@]}" > "$MK_ROOT/logs/server-$tag.log" 2>&1 & server=$!; fi ;;
    vllm)   taskset -c "$SERVER_CPUS" bash baselines/vllm/serve.sh "$model" "${vl_flags[@]}" > "$MK_ROOT/logs/server-$tag.log" 2>&1 & server=$! ;;
    mkserve:*) served="${MK_MODEL:-Qwen/Qwen3-8B}"
               taskset -c "$SERVER_CPUS" "$spy" -m mkserve serve "$served" --engine "${system#mkserve:}" --model-dir "$model" --port "$MK_PORT" --workdir "$MK_ROOT/mkserve" \
                 ${MKSERVE_CARD:+--card "$MKSERVE_CARD"} > "$MK_ROOT/logs/server-$tag.log" 2>&1 & server=$! ;;
  esac
  if wait_http "http://127.0.0.1:$MK_PORT/v1/models" "${MK_SERVER_WAIT:-1800}" $server; then
    stage "server $system up" ok $t
    grep -h -a -E "MK_FLAGS|MKSERVE" "$MK_ROOT/logs/server-$tag.log" | sort -u | cut -c1-260 | head -4
    taskset -c "$CLIENT_CPUS" "$MK_TOOLS_PY" scripts/demo_capture.py --port "$MK_PORT" --served "$served" --system "$system" --max-tokens "${DEMO_ANSWER_LEN:-400}" --out "$MK_ROOT/demo/$tag.json" 2>&1 | cut -c1-900
  else
    stage "server $system up" failed $t; tail -n 15 "$MK_ROOT/logs/server-$tag.log" | cut -c1-300
  fi
  kill "$server" 2>/dev/null; pkill -f "serve.sh|serve_image.sh" 2>/dev/null; pkill -f "vllm serve|vllm.entrypoints" 2>/dev/null; pkill -f "sglang.launch_server" 2>/dev/null; pkill -f "mkserve" 2>/dev/null
  sleep 8; pkill -9 -f "vllm|sglang|mkserve" 2>/dev/null; sleep 3
done
"$MK_TOOLS_PY" - "$MK_ROOT/demo" "${MK_MODEL:-Qwen/Qwen3-8B}" <<'PY'
import glob, json, subprocess, sys
d, model = sys.argv[1], sys.argv[2]
card = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.splitlines()
out = {"model": model, "card": f"{len(card)} x {card[0]}" if card else "?", "systems": [json.load(open(f)) for f in sorted(glob.glob(d + "/*.json")) if not f.endswith("demo.json")]}
json.dump(out, open(d + "/demo.json", "w"))
print("MK_DEMO_DONE", len(out["systems"]), "systems")
PY
store put "$MK_ROOT/demo" "$MK_S3/demo/${MK_JOB}" > /dev/null 2>&1; echo "MK_ENV demo stored rc=$?"
stage "demo capture" ok
