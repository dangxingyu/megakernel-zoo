#!/usr/bin/env bash
# bash scripts/mkserve_test.sh: start `python -m mkserve serve` for each engine in $MKSERVE_ENGINES on this pod's card and
# use it as a person would (scripts/mkserve_check.py). One MK_SERVE_TEST line per check.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MK_REPO"
# The whole output also goes to the object store (the job log service is not always readable).
mkdir -p "$MK_ROOT/report"; exec > >(tee "$MK_ROOT/report/out.log") 2>&1
env_report
mk_tools || exit 10
model="$(mk_model "${MK_MODEL:-Qwen/Qwen3-8B}")" || exit 11
bash scripts/cuda_toolkit.sh > "$MK_ROOT/logs/cuda-toolkit.log" 2>&1 || { tail -n 20 "$MK_ROOT/logs/cuda-toolkit.log" | cut -c1-300; stage "cuda toolkit" failed; exit 16; }
export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
py="$MK_ROOT/venvs/serve/bin/python"; t=$(date +%s)
{ uv venv -q "$MK_ROOT/venvs/serve" --python python3.12 && uv pip install -q --python "$py" torch numpy safetensors tokenizers aiohttp jinja2 huggingface_hub --torch-backend=auto; } \
  > "$MK_ROOT/logs/serve-venv.log" 2>&1 || { tail -n 20 "$MK_ROOT/logs/serve-venv.log" | cut -c1-300; stage "serve venv" failed $t; exit 12; }
stage "serve venv" ok $t
set +e +u
"$py" -m mkserve list
for engine in ${MKSERVE_ENGINES:-online-b offline-b}; do
  t=$(date +%s); log="$MK_ROOT/logs/mkserve-$engine.log"
  "$py" -m mkserve serve "${MK_MODEL:-Qwen/Qwen3-8B}" --engine "$engine" --model-dir "$model" --port "$MK_PORT" --workdir "$MK_ROOT/mkserve" ${MKSERVE_FLAGS:-} > "$log" 2>&1 & server=$!
  if wait_http "http://127.0.0.1:$MK_PORT/v1/models" 1500 $server; then
    stage "mkserve $engine up" ok $t
    "$py" scripts/mkserve_check.py --port "$MK_PORT" --engine "$engine" --model "${MK_MODEL:-Qwen/Qwen3-8B}" 2>&1 | cut -c1-1500
  else
    stage "mkserve $engine up" failed $t; tail -n 40 "$log" | cut -c1-400
  fi
  kill $server 2>/dev/null; wait $server 2>/dev/null; echo "MK_ENV mkserve $engine exit=$?"
  grep -a "MKSERVE\|mkserve:" "$log" | cut -c1-300 | head -12
  sleep 5; nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/MK_ENV gpu memory after stop: /'
done
stage "mkserve_test" ok
sleep 1; cp "$MK_ROOT"/logs/mkserve-*.log "$MK_ROOT"/logs/serve-venv.log "$MK_ROOT/report/" 2>/dev/null
store put "$MK_ROOT/report" "$MK_S3/mkserve/${MK_JOB:-test}" > /dev/null 2>&1; echo "MK_ENV report stored rc=$?"
