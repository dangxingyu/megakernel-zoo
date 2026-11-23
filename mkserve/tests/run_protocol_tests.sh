#!/usr/bin/env bash
# bash mkserve/tests/run_protocol_tests.sh [MODEL_DIR]: mkserve's APIs against a fake engine (CPU only, no GPU):
# the official OpenAI / Anthropic / Ollama SDKs (check_clients.py, in .venv-clients) and the rendered prompts against
# transformers' apply_chat_template (check_prompts.py).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; zoo="$(cd "$here/../.." && pwd)"
model="${1:-$HOME/models/Inkling-2-Mini}"; py="$zoo/.venv-mac/bin/python"; port="${PORT:-18690}"
tmp="$(mktemp -d)"; export MKSERVE_FAKE_SCRIPT="$tmp/script.json" MKSERVE_FAKE_LOG="$tmp/prompts.jsonl"
"$py" "$here/make_script.py" "$model" "$MKSERVE_FAKE_SCRIPT" || exit 2
cd "$zoo"
"$py" -m mkserve.server engine --workspace "$here/fake_engine" --model-dir "$model" --sock "$tmp/s" --mode native --max-context 32768 > "$tmp/engine.log" 2>&1 & e=$!
"$py" -m mkserve.server frontend --model-dir "$model" --sock "$tmp/s" --port "$port" --name inkling-2-mini --engine-name fake > "$tmp/front.log" 2>&1 & f=$!
trap 'kill $e $f 2>/dev/null; rm -rf "$tmp"' EXIT
for _ in $(seq 1 100); do curl -sf "http://127.0.0.1:$port/health" > /dev/null && break; sleep 0.2; done
"$zoo/.venv-clients/bin/python" "$here/check_clients.py" "http://127.0.0.1:$port"; rc=$?
"$py" "$here/check_prompts.py" "$model" "$MKSERVE_FAKE_LOG" || rc=1
for agent in claude codex opencode; do  # the launch commands name the served model and point at this server
  out="$("$py" -m mkserve launch "$agent" --url "http://127.0.0.1:$port" --workdir "$tmp" --print)"
  if [[ "$out" == *"127.0.0.1:$port"* || "$out" == *OPENCODE_CONFIG* ]] && { [ "$agent" = opencode ] || [[ "$out" == *inkling-2-mini* ]]; }; then
    echo "CHECK ok   launch $agent"; else echo "CHECK FAIL launch $agent: $out"; rc=1; fi
done
grep -q '"baseURL": "http://127.0.0.1:'"$port"'/v1"' "$tmp/opencode.json" && echo "CHECK ok   opencode config" || { echo "CHECK FAIL opencode config"; rc=1; }
[ $rc = 0 ] || { echo "--- front end log"; tail -30 "$tmp/front.log"; echo "--- engine log"; tail -10 "$tmp/engine.log"; }
exit $rc
