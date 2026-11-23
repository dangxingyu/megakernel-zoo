#!/usr/bin/env bash
# bash baselines/mac/bench.sh SYSTEM...: the bs=1 baselines of Inkling-2-Mini on this Mac, measured like the B200
# track-C baselines (eval/baselines_inkling-2-mini_b200_c.json): eval/client.py steady over HTTP, client clock, greedy,
# token-based rate (every SSE chunk weighted by the tokens it carries; a speculative server sends several per chunk).
#
#   synth    the evaluator's synthetic prompts, 64 tokens, 1,024 new tokens, ignore_eos
#   natural  eval/natural_prompts.py (64 tokens: prose, Python source, GSM8K), 1,024 new tokens, ignore_eos
#   chat     GSM8K questions through the chat template, up to 512 new tokens, until the model's own end token (no
#            ignore_eos: past its answer the model emits special tokens that stream as nothing, so the client's last
#            token time would come early); rate: end to end (all tokens / last token - send)
#
# Per server start: 1 warm-up + REPEATS (default 3) repeats of each workload; ROUNDS (default 2) server starts per
# system, in turns, so that warm-up and thermal drift fall on every system alike. SYSTEMS: see serve.sh.
# Results: OUT (default runs-mac/baselines-<date>)/SYSTEM-rROUND-WORKLOAD.json, and one MK_RESULT line each.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; zoo="$(cd "$here/../.." && pwd)"
py="$zoo/.venv-mac/bin/python"; models="${MODELS:-$HOME/models}"; tokdir="$models/Inkling-2-Mini"
out="${OUT:-$zoo/runs-mac/baselines-$(date +%Y%m%d-%H%M)}"; mkdir -p "$out/logs"
rounds="${ROUNDS:-2}"; repeats="${REPEATS:-3}"; port="${PORT:-18600}"; workloads="${WORKLOADS:-synth natural chat}"
[ -f "$out/../prompts/chat.json" ] || "$py" "$here/make_prompts.py" "$tokdir" "$out/../prompts"
prompts="$(cd "$out/../prompts" && pwd)"
sysctl -n machdep.cpu.brand_string hw.memsize > "$out/machine.txt"; sw_vers >> "$out/machine.txt"
pmset -g therm >> "$out/machine.txt" 2>/dev/null

wait_up() {  # until the server answers a 1-token completion (oMLX loads the model on first use)
  for _ in $(seq 1 600); do
    kill -0 "$1" 2>/dev/null || return 1
    curl -sf -m 600 "http://127.0.0.1:$port/v1/completions" -H 'Content-Type: application/json' \
      -d '{"model":"m","prompt":"Hi","max_tokens":1,"temperature":0}' > /dev/null && return 0
    sleep 2
  done; return 1
}

# a stopped bench must not leave its client running: it would go on sending to the next server on the port
trap 'pkill -P $$ 2>/dev/null; pkill -f "eval/client.py steady --base http://127.0.0.1:$port" 2>/dev/null' EXIT INT TERM

free_port() {  # stop whatever listens on the port (oMLX renames its process, so match by port, not by name)
  local p; p="$(lsof -t -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null)"; [ -n "$p" ] && kill $p 2>/dev/null
  for _ in $(seq 1 60); do lsof -nP -iTCP:"$port" -sTCP:LISTEN > /dev/null 2>&1 || return 0; sleep 1; done
  echo "port $port still busy" >&2; return 1
}

for round in $(seq 1 "$rounds"); do
  for sys in "$@"; do
    log="$out/logs/$sys-r$round.log"
    free_port || exit 3
    bash "$here/serve.sh" "$sys" "$port" > "$log" 2>&1 & pid=$!
    if ! wait_up "$pid"; then echo "MK_RESULT system=$sys round=$round ok=false (server did not come up: $log)"; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; continue; fi
    short=(); case "${sys%%-*}" in mlxlm|ollama) short=(--allow-short --min-tokens 64) ;; esac   # no ignore_eos in mlx_lm.server or Ollama
    for w in $workloads; do
      case "$w" in
        # text prompts for every system (oMLX's completions endpoint takes no token ids): the synthetic ones encode back
        # to exactly 64 tokens (the client checks), natural_text.json is natural.json decoded (48/48 encode back exactly)
        synth)   wl=(--text-prompts --prompt-tokens 64 --max-tokens 1024) ;;
        natural) wl=(--text-prompts --prompts-file "$prompts/natural_text.json" --max-tokens 1024) ;;
        chat)    wl=(--api chat --prompts-file "$prompts/chat.json" --max-tokens 512 --allow-short --min-tokens 16) ;;
      esac
      # chat: no ignore_eos (see the header); CHAT_PAUSE seconds between its requests (default 20, as measured)
      eos='{"ignore_eos": true}'; pause=1; [ "$w" = chat ] && { eos='{}'; pause="${CHAT_PAUSE:-20}"; }
      res="$out/$sys-r$round-$w.json"
      "$py" "$zoo/eval/client.py" steady --base "http://127.0.0.1:$port" --model m --model-dir "$tokdir" --batch 1 \
        --repeats "$repeats" --warmup 1 --pause "$pause" --extra-body "$eos" --save-text --label "mac-$sys-r$round-$w" \
        "${wl[@]}" ${short[@]+"${short[@]}"} --out "$res" > "$out/logs/$sys-r$round-$w.client.log" 2>&1
      "$py" - "$res" "$sys" "$round" "$w" <<'EOF'
import json, statistics as st, sys
path, sysname, rnd, w = sys.argv[1:]
try:
    d = json.load(open(path))
except Exception as e:
    print(f"MK_RESULT system={sysname} round={rnd} workload={w} ok=false ({e})"); raise SystemExit
def med(key):
    v = d.get(key)
    return v["median"] if isinstance(v, dict) else float("nan")
tps = med("decode_tok_s_by_tokens")
print(f"MK_RESULT system={sysname} round={rnd} workload={w} tok_s={tps:.1f} us_per_tok={1e6/tps:.0f} "
      f"chunk_tok_s={med('decode_tok_s'):.1f} tok_per_chunk={d.get('tokens_per_chunk') or 0:.2f} "
      f"ttft_s={d.get('ttft_median_s') or float('nan'):.3f} clean={d.get('clean_repeats')}/{d.get('repeats')}")
EOF
    done
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; free_port; sleep 5
  done
done | tee -a "$out/results.txt"
