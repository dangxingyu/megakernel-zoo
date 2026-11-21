#!/usr/bin/env bash
# bash scripts/compare.sh: the like-for-like comparison. On ONE card, in turns, every system is served over HTTP
# and measured by the same client (eval/client.py steady: prompt 64, 1024 new tokens, greedy, one SSE chunk per
# token, client clock):
#   sglang        SGLang $SGLANG_VERSION (default 0.5.18) with --enable-torch-compile: the fastest general engine found
#   vllm          vLLM (pinned in baselines/vllm/install.sh), default flags
#   ours:NAME     the engine source rescore_inbox/NAME.tgz behind eval/serve_engine.py (front end + engine process)
#   cohere        Cohere's megakernel server (baselines/cohere; North Mini Code only, x86 H100 only). Its completions
#                 endpoint does not stream, so it is measured through the chat endpoint and may stop at an end token
#                 (--api chat --allow-short): steady workload only
# COMPARE_WORKLOAD=replay measures the Mooncake trace replay instead (FAST'25, Kimi tool & agent trace; open loop at
# COMPARE_RATES req/s, default "1 2"; REPLAY_S seconds of arrivals, default 300, plus a drain): one server start per
# (system, rate), so every replay begins with an empty prefix cache. ours:NAME must then have the online contract
# (submit / poll). One MK_REPLAY line per (system, rate, round), label compare-SYSTEM-qRATE-rROUND.
# Environment: COMPARE_SYSTEMS (space separated, default: every ours:* in rescore_inbox, then sglang vllm),
# COMPARE_ROUNDS (default 3), COMPARE_BATCHES (default "1 8"), COMPARE_REPEATS (default 3 per server start).
# COMPARE_IMAGES=1 serves sglang and vllm from their vendors' images instead of pip (scripts/ink/img.sh; models whose
# support ships only as an image, e.g. Inkling-2-Small: COMPARE_SGL_IMAGE / COMPARE_VLLM_IMAGE = REPO@sha256:..., the
# Inkling-2-Small day-0 images by default; COMPARE_SG_FLAGS / COMPARE_VL_FLAGS are then the whole flag list, and
# COMPARE_SG_ENV the SGLang recipe's environment). MK_TARGET=node: a whole-node engine; the server gets most of the
# node's cores. Each round starts every server afresh and begins with a different system, so that order, warm-up and drift of the
# node fall on all systems alike. One MK_RESULT line per (system, round, batch), label compare-SYSTEM-rROUND.
# COMPARE_CELLS="P:N:B:MODE ..." replaces the one steady workload by a matrix on the same server instance: prompt of P
# tokens, N new tokens, B concurrent streams, MODE g (greedy) or s (sampled: temperature 0.7, top_p 0.8, top_k 20); G and S
# are the same with prompts of natural text (eval/natural_prompts.py: prose, code and word problems, P tokens each)
# instead of the synthetic ones. ours:* serves greedy only and skips the sampled cells.
# Label rel-SYSTEM-pP-nN-bB-MODE-rROUND; a cell a system cannot serve gives one MK_CELL line with ok=false (and the
# server is started again if the cell killed it). COMPARE_CELL_REPEATS (default 2), COMPARE_CELL_TIMEOUT seconds per
# cell (default 900). COMPARE_OURS_FLAGS replaces serve_engine.py's "--max-batch 8 --max-context 2048".
# mkserve:ENGINE is our engine behind `python -m mkserve serve` (step mode, the path that samples).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$MK_REPO"
env_report
mk_tools || exit 10
uv pip install -q --python "$MK_TOOLS_PY" nvidia-ml-py
model="$(mk_model "${MK_MODEL:-Qwen/Qwen3-8B}")" || exit 11
bash scripts/cuda_toolkit.sh > "$MK_ROOT/logs/cuda-toolkit.log" 2>&1 || { tail -n 20 "$MK_ROOT/logs/cuda-toolkit.log" | cut -c1-300; stage "cuda toolkit" failed; exit 16; }
grep MK_ENV "$MK_ROOT/logs/cuda-toolkit.log"; export CUDA_HOME=/usr/local/cuda PATH="/usr/local/cuda/bin:$PATH"
set +e +u; set -o pipefail

if [ -n "${COMPARE_SYSTEMS:-}" ]; then read -r -a systems <<< "$COMPARE_SYSTEMS"; else
  systems=(); for src in rescore_inbox/*.tgz; do [ -f "$src" ] && systems+=("ours:$(basename "$src" .tgz)"); done; systems+=(sglang vllm)
fi
rounds="${COMPARE_ROUNDS:-3}"; batches="${COMPARE_BATCHES:-1 8}"; repeats="${COMPARE_REPEATS:-3}"
workload="${COMPARE_WORKLOAD:-steady}"; rates="${COMPARE_RATES:-1 2}"
echo "MK_ENV compare workload=$workload systems=${systems[*]} rounds=$rounds batches=$batches repeats=$repeats rates=$rates"
# COMPARE_SG_FLAGS / COMPARE_VL_FLAGS: further serve flags of the two general engines (for example --quantization fp8).
read -r -a sg_more <<< "${COMPARE_SG_FLAGS:-}"; read -r -a vl_more <<< "${COMPARE_VL_FLAGS:-}"
# COMPARE_IMAGES=1 also offers vllm-spec (vLLM's flags plus COMPARE_VL_SPEC) and sglang-spec from the image (SGLang's
# flags plus COMPARE_SG_SPEC, environment COMPARE_SG_SPEC_ENV); COMPARE_VLLM_WHEEL=1 serves vLLM PR #69028's wheel from
# the store over the nightly image (scripts/mini/vllm_pr_build.sh) instead of COMPARE_VLLM_IMAGE.
# COMPARE_SG_BASE replaces SGLang's base flags (default --enable-torch-compile, the fastest setting of 0.5.18; broken on
# 0.5.23). System `sglang-spec` is the same SGLang with COMPARE_SG_SPEC added (its native speculative decoding).
read -r -a sg_base <<< "${COMPARE_SG_BASE---enable-torch-compile}"; read -r -a sg_spec <<< "${COMPARE_SG_SPEC:-}"
sg_flags=("${sg_base[@]}" "${sg_more[@]}"); vl_flags=("${vl_more[@]}"); read -r -a ours_flags <<< "${COMPARE_OURS_FLAGS:---max-batch 8 --max-context 2048}"; cases="steady"
cells="${COMPARE_CELLS:-}"; spy="$MK_ROOT/venvs/serve/bin/python"
images="${COMPARE_IMAGES:-0}"
if [ "$images" = 1 ]; then
  source scripts/ink/img.sh
  sg_flags=("${sg_more[@]}")  # the recipe's flags as given, nothing added
  SGL_IMAGE="${COMPARE_SGL_IMAGE:-lmsysorg/sglang@sha256:198af079c599cbf37ad371faca16c988a71cea771b3c51fbe21d018f6e400c9e}"
  VLLM_IMAGE="${COMPARE_VLLM_IMAGE:-vllm/vllm-openai@sha256:bc9f13e3fca71bea6a8603cc880fc705b6ebe3c21b438371d744ace7de07f0e7}"
fi
if [ "$workload" = replay ]; then
  commit="aaa49180771f"; trace_rps=6.67; trace="$MK_ROOT/run/toolagent_trace.jsonl"
  curl -fsSL -o "$trace" "https://raw.githubusercontent.com/kvcache-ai/Mooncake/$commit/FAST25-release/traces/toolagent_trace.jsonl" || exit 16
  echo "MK_ENV trace=toolagent commit=${commit:0:8} records=$(wc -l < "$trace") sha256=$(sha256sum "$trace" | cut -c1-16)"
  # Both general engines report how many prompt tokens came from the prefix cache only when asked to.
  sg_flags+=(--enable-cache-report); vl_flags+=(--enable-prompt-tokens-details); ours_flags=(--max-batch 64 --max-context 34816); cases="$rates"
fi

eval_ready=0
for system in "${systems[@]}"; do
  t=$(date +%s)
  case "$system" in
    sglang-spec) if [ "$images" = 1 ]; then [ -x "$MK_ROOT/img/sglang/bin/bash" ] && [ "${sgl_ready:-0}" = 1 ] || { img_pull sglang "$SGL_IMAGE" > "$MK_ROOT/logs/install-sglang.log" 2>&1 || { tail -n 20 "$MK_ROOT/logs/install-sglang.log"; exit 12; }; grep MK_ENV "$MK_ROOT/logs/install-sglang.log"; }
              sgl_ready=1; stage "prepare $system" ok $t; continue; fi
                 [ -x "$MK_ROOT/venvs/sglang/bin/python" ] || { SGLANG_VERSION="${SGLANG_VERSION:-0.5.18}" bash baselines/sglang/install.sh > "$MK_ROOT/logs/install-sglang.log" 2>&1 || { tail -n 30 "$MK_ROOT/logs/install-sglang.log" | cut -c1-300; exit 12; }; grep MK_ENV "$MK_ROOT/logs/install-sglang.log"; } ;;
    sglang) if [ "$images" = 1 ]; then [ -x "$MK_ROOT/img/sglang/bin/bash" ] && [ "${sgl_ready:-0}" = 1 ] || { img_pull sglang "$SGL_IMAGE" > "$MK_ROOT/logs/install-sglang.log" 2>&1 || { tail -n 20 "$MK_ROOT/logs/install-sglang.log"; exit 12; }; grep MK_ENV "$MK_ROOT/logs/install-sglang.log"; }
              sgl_ready=1; stage "prepare $system" ok $t; continue; fi
            SGLANG_VERSION="${SGLANG_VERSION:-0.5.18}" bash baselines/sglang/install.sh > "$MK_ROOT/logs/install-sglang.log" 2>&1 \
              || { tail -n 30 "$MK_ROOT/logs/install-sglang.log" | cut -c1-300; stage "install sglang" failed; exit 12; }
            grep MK_ENV "$MK_ROOT/logs/install-sglang.log" ;;
    vllm|vllm-spec) if [ "$images" = 1 ]; then
              if [ -x "$MK_ROOT/img/vllm/bin/bash" ] && [ "${vllm_ready:-0}" = 1 ]; then stage "prepare $system" ok $t; continue; fi
              if [ "${COMPARE_VLLM_WHEEL:-0}" = 1 ]; then  # vLLM PR #69028's wheel (store) over the nightly image
                VLLM_PR_THEN_MEASURE=0 bash scripts/mini/vllm_pr_build.sh > "$MK_ROOT/logs/install-vllm.log" 2>&1
                grep -E "MK_ENV|MK_STAGE" "$MK_ROOT/logs/install-vllm.log" | cut -c1-300
                [ -x "$MK_ROOT/img/vllm/bin/bash" ] || { tail -n 20 "$MK_ROOT/logs/install-vllm.log"; exit 12; }
                export IMG_EXTRA_ENV="CUDA_HOME=/usr/local/cuda"   # as vllm_pr_build.sh's own measurement: FlashInfer JIT wants nvcc
              else img_pull vllm "$VLLM_IMAGE" > "$MK_ROOT/logs/install-vllm.log" 2>&1 || { tail -n 20 "$MK_ROOT/logs/install-vllm.log"; exit 12; }
                grep MK_ENV "$MK_ROOT/logs/install-vllm.log"; fi
              vllm_ready=1; stage "prepare $system" ok $t; continue; fi
            [ "$system" = vllm ] || { echo "vllm-spec needs COMPARE_IMAGES=1"; exit 18; }
            bash baselines/vllm/install.sh > "$MK_ROOT/logs/install-vllm.log" 2>&1 \
              || { tail -n 30 "$MK_ROOT/logs/install-vllm.log" | cut -c1-300; stage "install vllm" failed; exit 12; }
            grep MK_ENV "$MK_ROOT/logs/install-vllm.log" ;;
    cohere) bash baselines/cohere/install.sh > "$MK_ROOT/logs/install-cohere.log" 2>&1 \
              || { tail -n 30 "$MK_ROOT/logs/install-cohere.log" | cut -c1-300; stage "install cohere" failed; exit 12; }
            grep MK_ENV "$MK_ROOT/logs/install-cohere.log" ;;
    ours:*) name="${system#ours:}"; ws="$MK_ROOT/compare/$name"
            if [ "$eval_ready" = 0 ]; then bash scripts/eval_setup.sh 2>&1 | grep -E "MK_ENV eval |rror|mismatch" | cut -c1-300; eval_ready=1; fi
            rm -rf "$ws"; mkdir -p "$ws"; tar xzf "rescore_inbox/$name.tgz" -C "$ws" || exit 17
            (cd "$ws" && export PATH="$MK_ROOT/venvs/eval/bin:$PATH" && { [ ! -f engine/build.sh ] || bash engine/build.sh; }) > "$MK_ROOT/logs/build-$name.log" 2>&1 \
              || { tail -n 20 "$MK_ROOT/logs/build-$name.log" | cut -c1-300; stage "build $name" failed; exit 17; }
            echo "MK_ENV ours $name sha256=$(sha256sum "rescore_inbox/$name.tgz" | cut -c1-16)" ;;
    mkserve:*) [ -x "$spy" ] || { uv venv -q "$MK_ROOT/venvs/serve" --python python3.12 && uv pip install -q --python "$spy" torch numpy safetensors tokenizers aiohttp jinja2 huggingface_hub --torch-backend=auto; } \
                 > "$MK_ROOT/logs/serve-venv.log" 2>&1 || { tail -n 20 "$MK_ROOT/logs/serve-venv.log" | cut -c1-300; stage "serve venv" failed $t; exit 12; } ;;
    *) echo "unknown system $system"; exit 18 ;;
  esac
  stage "prepare $system" ok $t
done
cpu_args=()
if [ "${MK_TARGET:-card}" = node ]; then  # a whole-node engine (a process per card): most of the node's physical cores
  phys=$(lscpu -p=core,socket 2>/dev/null | grep -v '^#' | sort -u | wc -l); cpu_args=(--server $(( phys - 12 )) --client 4)
fi
picked="$(python3 scripts/pick_cpus.py "${cpu_args[@]}")" || { echo "MK_ENV $picked"; exit 15; }
echo "MK_ENV $picked"; eval "$picked"

server=""; worker=""
start_server() {  # $1 = system
  local tag="${1//:/-}"
  # another job on the same host (shared network) may already serve this port: its server would answer our client
  if curl -s -m 3 -o /dev/null "http://127.0.0.1:$MK_PORT/v1/models"; then
    echo "MK_ENV port $MK_PORT is already served by another process on this host: no measurement"; return 1
  fi
  case "$1" in
    sglang) if [ "$images" = 1 ]; then SGL_IMAGE_ENV="${COMPARE_SG_ENV:-}" taskset -c "$SERVER_CPUS" bash baselines/sglang/serve_image.sh "$model" "${sg_flags[@]}" > "$MK_ROOT/logs/server-$tag.log" 2>&1 & server=$!
            else taskset -c "$SERVER_CPUS" bash baselines/sglang/serve.sh "$model" "${sg_flags[@]}" > "$MK_ROOT/logs/server-$tag.log" 2>&1 & server=$!; fi ;;
    vllm)   if [ "$images" = 1 ]; then taskset -c "$SERVER_CPUS" bash baselines/vllm/serve_image.sh "$model" "${vl_flags[@]}" > "$MK_ROOT/logs/server-$tag.log" 2>&1 & server=$!
            else taskset -c "$SERVER_CPUS" bash baselines/vllm/serve.sh "$model" "${vl_flags[@]}" > "$MK_ROOT/logs/server-$tag.log" 2>&1 & server=$!; fi ;;
    cohere) taskset -c "$SERVER_CPUS" bash baselines/cohere/serve.sh "$model" > "$MK_ROOT/logs/server-$tag.log" 2>&1 & server=$! ;;
    sglang-spec) if [ "$images" = 1 ]; then SGL_IMAGE_ENV="${COMPARE_SG_SPEC_ENV:-${COMPARE_SG_ENV:-}}" taskset -c "$SERVER_CPUS" bash baselines/sglang/serve_image.sh "$model" "${sg_flags[@]}" "${sg_spec[@]}" > "$MK_ROOT/logs/server-$tag.log" 2>&1 & server=$!
                 else taskset -c "$SERVER_CPUS" bash baselines/sglang/serve.sh "$model" "${sg_flags[@]}" "${sg_spec[@]}" > "$MK_ROOT/logs/server-$tag.log" 2>&1 & server=$!; fi ;;
    vllm-spec) # COMPARE_VL_SPEC (e.g. --speculative-config {...}): words with a brace are single-quoted, since
               # serve_image.sh joins its arguments into one shell string (as scripts/mini/mtp_baseline.sh)
               local q; q="$(python3 -c 'import sys; print(" ".join(chr(39) + a + chr(39) if "{" in a else a for a in sys.argv[1].split()))' "${vl_flags[*]} ${COMPARE_VL_SPEC:-}")"
               taskset -c "$SERVER_CPUS" bash baselines/vllm/serve_image.sh "$model" $q > "$MK_ROOT/logs/server-$tag.log" 2>&1 & server=$! ;;
    mkserve:*) taskset -c "$SERVER_CPUS" "$spy" -m mkserve serve "${MK_MODEL:-Qwen/Qwen3-8B}" --engine "${1#mkserve:}" --mode step --model-dir "$model" \
                 --port "$MK_PORT" --workdir "$MK_ROOT/mkserve" ${MKSERVE_CARD:+--card "$MKSERVE_CARD"} > "$MK_ROOT/logs/server-$tag.log" 2>&1 & server=$! ;;
    ours:*) local name="${1#ours:}"; local ws="$MK_ROOT/compare/$name" sock="$MK_ROOT/run/ours-$name.sock"; rm -f "$sock"
            (cd "$ws" && PATH="$MK_ROOT/venvs/eval/bin:$PATH" exec taskset -c "$SERVER_CPUS" "$MK_ROOT/venvs/eval/bin/python" "$MK_REPO/eval/serve_engine.py" engine \
               --workspace "$ws" --model-dir "$model" --sock "$sock" "${ours_flags[@]}") > "$MK_ROOT/logs/engine-$tag.log" 2>&1 & worker=$!
            taskset -c "$SERVER_CPUS" "$MK_TOOLS_PY" eval/serve_engine.py frontend --model-dir "$model" --sock "$sock" --port "$MK_PORT" > "$MK_ROOT/logs/server-$tag.log" 2>&1 & server=$!
            echo "MK_FLAGS serve_engine.py engine ${ours_flags[*]} + frontend; source $name" ;;
  esac
  wait_http "http://127.0.0.1:$MK_PORT/v1/models" "${MK_SERVER_WAIT:-1200}" $server && return 0
  tail -n 15 "$MK_ROOT/logs/server-$tag.log" "$MK_ROOT/logs/engine-$tag.log" 2>/dev/null | cut -c1-300
  # the root cause is usually far above the last traceback (an engine process that died earlier)
  grep -h -E "Error|ERROR|larger than|out of memory|not supported" "$MK_ROOT/logs/server-$tag.log" 2>/dev/null | grep -v "Engine core initialization failed" | head -n 8 | cut -c1-500 | sed 's/^/MK_SERVER_ERROR /'
  return 1
}
stop_server() {
  pkill -f "serve.sh|serve_image.sh" 2>/dev/null; pkill -f "vllm serve|vllm.entrypoints" 2>/dev/null; pkill -f "sglang.launch_server" 2>/dev/null; pkill -f "serve_engine.py" 2>/dev/null
  pkill -f "serving/server.py" 2>/dev/null; pkill -f "mkserve serve" 2>/dev/null
  sleep 8; pkill -9 -f "vllm|sglang|serve_engine.py|serving/server.py|mkserve serve" 2>/dev/null; sleep 3; server=""; worker=""
}

n=${#systems[@]}
for round in $(seq 1 "$rounds"); do
  for k in $(seq 0 $((n - 1))); do
    system="${systems[$(( (k + round - 1) % n ))]}"; tag="${system//:/-}"
    for case in $cases; do
      t=$(date +%s)
      if ! start_server "$system"; then stage "server $system round $round" failed $t; stop_server; continue; fi
      grep -h -E "MK_FLAGS|MK_SERVE" "$MK_ROOT/logs/server-$tag.log" "$MK_ROOT/logs/engine-$tag.log" 2>/dev/null | sort -u | cut -c1-300
      stage "server $system round $round up" ok $t
      if [ "$workload" = replay ]; then
        scale=$(python3 -c "print($case / $trace_rps)")
        taskset -c "$CLIENT_CPUS" "$MK_TOOLS_PY" eval/client.py replay --base "http://127.0.0.1:$MK_PORT" --model m --model-dir "$model" \
          --trace "$trace" --rate-scale "$scale" --duration-s "${REPLAY_S:-300}" --drain-s 90 --max-input-tokens 32768 --read-timeout 900 \
          --extra-body '{"ignore_eos": true}' --nvml --label "compare-$tag-q$case-r$round" --out "$MK_ROOT/run/compare-$tag-q$case-r$round.json" 2>&1 \
          | grep -E "MK_REPLAY|rror" | cut -c1-2500
      else
        served=m; how=(--extra-body '{"ignore_eos": true}')
        if [ "$system" = cohere ]; then
          served="$(curl -s "http://127.0.0.1:$MK_PORT/v1/models" | "$MK_TOOLS_PY" -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
          how=(--api chat --allow-short)
        fi
        [ -n "$cells" ] || for b in $batches; do
          taskset -c "$CLIENT_CPUS" "$MK_TOOLS_PY" eval/client.py steady --base "http://127.0.0.1:$MK_PORT" --model "$served" --model-dir "$model" --batch "$b" \
            --prompt-tokens 64 --max-tokens 1024 --repeats "$repeats" --warmup 1 --nvml "${how[@]}" \
            --label "compare-$tag-r$round" --out "$MK_ROOT/run/compare-$tag-r$round-b$b.json" 2>&1 | grep -E "MK_RESULT|clean\": false|rror" | cut -c1-1500
        done
        for cell in $cells; do
          IFS=: read -r cp cn cb cm <<< "$cell"; tc=$(date +%s); out="$MK_ROOT/run/rel-$tag-p$cp-n$cn-b$cb-$cm-r$round.json"
          case "$system:$cm" in ours:*:s|ours:*:S) continue ;; esac   # serve_engine.py decodes greedily whatever the request says
          body='{"ignore_eos": true, "temperature": 0}'; case "$cm" in s|S) body='{"ignore_eos": true, "temperature": 0.7, "top_p": 0.8, "top_k": 20}' ;; esac
          nat=(); case "$cm" in G|S) pf="$MK_ROOT/run/natural-p$cp.json"
            [ -s "$pf" ] || "$MK_TOOLS_PY" eval/natural_prompts.py "$model" "$pf" "$cp" 48 || { echo "MK_CELL_WHY no natural prompts"; continue; }
            nat=(--prompts-file "$pf") ;; esac
          timeout "${COMPARE_CELL_TIMEOUT:-900}" taskset -c "$CLIENT_CPUS" "$MK_TOOLS_PY" eval/client.py steady --base "http://127.0.0.1:$MK_PORT" --model "$served" --model-dir "$model" \
            --batch "$cb" --prompt-tokens "$cp" --max-tokens "$cn" --repeats "${COMPARE_CELL_REPEATS:-2}" --warmup 1 --nvml --read-timeout "${COMPARE_CELL_TIMEOUT:-900}" "${nat[@]}" \
            --extra-body "$body" --label "rel-$tag-p$cp-n$cn-b$cb-$cm-r$round" --out "$out" > "$MK_ROOT/logs/cell.log" 2>&1; rc=$?
          grep -E "MK_RESULT" "$MK_ROOT/logs/cell.log" | cut -c1-1500
          ok=false; [ $rc = 0 ] && grep -q "MK_RESULT" "$MK_ROOT/logs/cell.log" && ! grep -q 'clean": false' "$MK_ROOT/logs/cell.log" && ok=true
          alive=true; kill -0 "$server" 2>/dev/null || alive=false
          [ -z "$worker" ] || kill -0 "$worker" 2>/dev/null || alive=false   # our engine process behind the front end
          why=""; [ $ok = true ] || why="$(grep -E 'rror|clean": false' "$MK_ROOT/logs/cell.log" | tail -n 2 | tr -d '"\\' | tr '\n' ' ' | cut -c1-300)"
          echo "MK_CELL {\"system\": \"$system\", \"round\": $round, \"prompt\": $cp, \"new\": $cn, \"batch\": $cb, \"mode\": \"$cm\", \"ok\": $ok, \"rc\": $rc, \"server_alive\": $alive, \"seconds\": $(( $(date +%s) - tc )), \"why\": \"$why\"}"
          if [ $ok = false ]; then  # a failed cell may have killed or hung the server or left it in a bad state: a fresh one for the next cell
            echo "MK_CELL_WHY $(grep -h -E "Error|error:|watchdog|assert" "$MK_ROOT/logs/engine-$tag.log" "$MK_ROOT/logs/server-$tag.log" 2>/dev/null | tail -n 2 | tr '\n' ' ' | cut -c1-400)"
            tail -n 6 "$MK_ROOT/logs/engine-$tag.log" "$MK_ROOT/logs/server-$tag.log" 2>/dev/null | cut -c1-300
            stop_server; start_server "$system" || { stage "server $system round $round restart" failed; break; }
          fi
        done
      fi
      kill -0 "$server" 2>/dev/null || { echo "MK_ENV server $system died during round $round"; tail -n 12 "$MK_ROOT/logs/engine-$tag.log" 2>/dev/null | cut -c1-300; }
      stop_server
    done
  done
done
rm -f "$MK_ROOT"/run/*.sock "$MK_ROOT"/run/*_trace.jsonl
mkdir -p "$MK_ROOT/run/logs" && cp "$MK_ROOT"/logs/*.log "$MK_ROOT/run/logs/" 2>/dev/null
store put "$MK_ROOT/run" "$MK_S3/runs/${MK_JOB}" "$MK_S3" 2>&1 | tail -n 1 | cut -c1-200
stage "compare ab" ok
