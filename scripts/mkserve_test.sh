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
[ -n "${MKSERVE_V2:-}" ] && { echo "MK_V2 doctor"; "$py" -m mkserve doctor "${MK_MODEL:-Qwen/Qwen3-8B}" 2>&1 | tail -14; }
for engine in ${MKSERVE_ENGINES:-online-b offline-b}; do
  t=$(date +%s); log="$MK_ROOT/logs/mkserve-$engine.log"
  "$py" -m mkserve serve "${MK_MODEL:-Qwen/Qwen3-8B}" --engine "$engine" --model-dir "$model" --port "$MK_PORT" --workdir "$MK_ROOT/mkserve" ${MKSERVE_FLAGS:-} > "$log" 2>&1 & server=$!
  if wait_http "http://127.0.0.1:$MK_PORT/v1/models" 1500 $server; then
    stage "mkserve $engine up" ok $t
    "$py" scripts/mkserve_check.py --port "$MK_PORT" --engine "$engine" --model "${MK_MODEL:-Qwen/Qwen3-8B}" 2>&1 | cut -c1-1500
    if [ -n "${MKSERVE_V2:-}" ]; then   # the v2 commands against this server
      echo "MK_V2 bench"; "$py" -m mkserve bench --url "http://127.0.0.1:$MK_PORT" --batch 1 8 --rounds 1 --max-tokens 256 2>&1 | tail -5
      echo "MK_V2 chat"; printf 'What is 17 times 3? Answer in one sentence.\n/exit\n' | "$py" -m mkserve chat --url "http://127.0.0.1:$MK_PORT" --max-tokens 64 2>&1 | tail -4
      echo "MK_V2 metrics"; curl -s "http://127.0.0.1:$MK_PORT/metrics" | grep -v '^#' | head -12
    fi
  else
    stage "mkserve $engine up" failed $t; tail -n 40 "$log" | cut -c1-400
  fi
  kill $server 2>/dev/null; wait $server 2>/dev/null; echo "MK_ENV mkserve $engine exit=$?"
  grep -a "MKSERVE\|mkserve:" "$log" | cut -c1-300 | head -12
  sleep 5; nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/MK_ENV gpu memory after stop: /'
done
if [ -n "${MKSERVE_V2:-}" ]; then   # --api-key and --served-model-name on a second start
  engine=${MKSERVE_ENGINES%% *}; log="$MK_ROOT/logs/mkserve-key.log"
  "$py" -m mkserve serve "${MK_MODEL:-Qwen/Qwen3-8B}" --engine "$engine" --model-dir "$model" --port "$MK_PORT" --workdir "$MK_ROOT/mkserve" --api-key sekrit --served-model-name mk > "$log" 2>&1 & server=$!
  if wait_http "http://127.0.0.1:$MK_PORT/health" 1500 $server; then
    echo "MK_V2 no key: $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$MK_PORT/v1/models)"
    echo "MK_V2 with key: $(curl -s -H 'Authorization: Bearer sekrit' http://127.0.0.1:$MK_PORT/v1/models | cut -c1-200)"
    echo "MK_V2 bench with key"; "$py" -m mkserve bench --url "http://127.0.0.1:$MK_PORT" --api-key sekrit --batch 1 --rounds 1 --max-tokens 64 2>&1 | tail -2
  else
    echo "MK_V2 key server did not come up"; tail -n 20 "$log" | cut -c1-300
  fi
  kill $server 2>/dev/null; wait $server 2>/dev/null
fi
stage "mkserve_test" ok
sleep 1; cp "$MK_ROOT"/logs/mkserve-*.log "$MK_ROOT"/logs/serve-venv.log "$MK_ROOT/report/" 2>/dev/null
store put "$MK_ROOT/report" "$MK_S3/mkserve/${MK_JOB:-test}" > /dev/null 2>&1; echo "MK_ENV report stored rc=$?"
