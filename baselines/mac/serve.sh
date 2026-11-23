#!/usr/bin/env bash
# bash baselines/mac/serve.sh SYSTEM [PORT]: serve Inkling-2-Mini on this Mac with one general engine, OpenAI API on PORT
# (default 18600), one sequence at a time. The process runs in the foreground; bench.sh starts and stops it.
#
#   llamacpp-q4      llama.cpp (Metal), bartowski Q4_K_M GGUF, no speculation
#   llamacpp-q4-ds   llama.cpp #32218 with the checkpoint's DSpark draft head (BF16 GGUF), --spec-type draft-dspark
#   llamacpp-q4-pr   llama.cpp #32218 without a draft (the -ds system's own plain decode)
#   llamacpp-q8      llama.cpp, ggml-org Q8_0 GGUF, no speculation
#   llamacpp-q8-ds   the same plus DSpark
#   mlxlm-native     mlx-lm (ml-explore/mlx-lm#2086) on the checkpoint itself (NVFP4/MXFP8 as stored), no speculation
#   omlx-oq4e        oMLX v0.8.3 on mlx-community/Inkling-2-Mini-oQ4e-mtp, no speculation
#   omlx-oq4e-ds     the same with its DSpark drafting (mtp_enabled: Lightning MTP on the bundled mtp.safetensors)
#   ollama-q4        Ollama (Homebrew), bartowski Q4_K_M GGUF (its bundled llama-server runner), no speculation
#   ollama-mlx       Ollama's MLX runner on the checkpoint's safetensors as stored (no -q), no speculation
#   ollama-mlx-ds    the same with DRAFT = the checkpoint's mtp.safetensors (Ollama's speculative decoder)
#   Ollama systems need their model "m" made once: bash baselines/mac/setup_ollama.sh SYSTEM
#
# Paths: MODELS (default ~/models), LLAMA_CPP (default ~/src/llama.cpp: master), LLAMA_CPP_DS (default
# ~/src/llama.cpp-dspark: ggml-org/llama.cpp#32218, branch pwilkin/inkling2-dspark, the Inkling 2 DSpark head, which
# master does not have), OMLX (default ~/src/omlx).
# DS_N: draft block length for the -ds systems (default 7, the vLLM recipe's num_speculative_tokens).
set -euo pipefail
sys="${1:?system}"; port="${2:-18600}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; zoo="$(cd "$here/../.." && pwd)"
models="${MODELS:-$HOME/models}"; lcpp="${LLAMA_CPP:-$HOME/src/llama.cpp}"; lcpp_ds="${LLAMA_CPP_DS:-$HOME/src/llama.cpp-dspark}"; omlx="${OMLX:-$HOME/src/omlx}"
ds_n="${DS_N:-7}"; ctx="${CTX:-8192}"

llama() {  # llama.cpp server: all layers on the GPU, flash attention, one slot, no prompt cache reuse between requests
  exec "$lcpp/build/bin/llama-server" --host 127.0.0.1 --port "$port" -m "$1" -ngl 999 -fa on -c "$ctx" \
    --parallel 1 --no-cache-prompt --no-webui --jinja --alias m "${@:2}"
}
# The draft: the checkpoint's mtp.safetensors converted by #32218's own converter (baselines/mac/README.md):
#   convert_hf_to_gguf.py ~/models/Inkling-2-Mini --dspark --outtype bf16 --outfile Inkling-2-Mini-DSpark-bf16.gguf
# (Unsloth's dspark-Inkling-2-Mini-Q8_0.gguf loads in neither master nor #32218: no markov_w2, padded vocabulary.)
ds=(-md "$models/gguf-dspark-pr/Inkling-2-Mini-DSpark-bf16.gguf" -ngld 999 --spec-type draft-dspark --spec-draft-n-max "$ds_n")

case "$sys" in
  # bartowski's Q4_K_M: arch inkling2_loop (24 stored layers), what llama.cpp master loads. Unsloth's UD GGUFs say
  # arch inkling2 with 48 blocks, which master refuses (tensor blk.24.attn_q.weight not found).
  llamacpp-q4)    llama "$models/gguf-bartowski/Inkling-2-Mini-Q4_K_M.gguf" ;;
  llamacpp-q4-ds) lcpp="$lcpp_ds"; llama "$models/gguf-bartowski/Inkling-2-Mini-Q4_K_M.gguf" "${ds[@]}" ;;
  llamacpp-q4-pr) lcpp="$lcpp_ds"; llama "$models/gguf-bartowski/Inkling-2-Mini-Q4_K_M.gguf" ;;   # #32218 without a draft
  llamacpp-q8)    llama "$models/gguf-ggml-org/Inkling-2-Mini-Q8_0.gguf" ;;
  llamacpp-q8-ds) lcpp="$lcpp_ds"; llama "$models/gguf-ggml-org/Inkling-2-Mini-Q8_0.gguf" "${ds[@]}" ;;
  mlxlm-native)
    # mlx_lm.server resolves the request's "model" as a path or a Hugging Face id: run it next to a link named m
    root="$models/mlxlm-root"; mkdir -p "$root"; ln -sfn "$models/Inkling-2-Mini" "$root/m"; cd "$root"
    exec "$zoo/.venv-mac/bin/mlx_lm.server" --host 127.0.0.1 --port "$port" --model m ;;
  omlx-oq4e|omlx-oq4e-ds)
    # oMLX serves every model folder under --model-dir; a private folder holds just this one, and its per-model
    # settings switch DSpark drafting on or off.
    root="$models/omlx-root-$sys"; mkdir -p "$root"; ln -sfn "$models/mlx-oq4e-mtp" "$root/m"
    on=false; [ "$sys" = omlx-oq4e-ds ] && on=true
    export OMLX_BASE_PATH="$root/.omlx"; mkdir -p "$OMLX_BASE_PATH"
    # Inkling 2: mtp_enabled loads the bundled mtp.safetensors as a DSpark drafter (engine_pool.py); depth adaptive
    # (oMLX's default) unless OMLX_DEPTH fixes it.
    fixed=null; [ -n "${OMLX_DEPTH:-}" ] && fixed="$OMLX_DEPTH"
    printf '{"version": 3, "models": {"m": {"mtp_enabled": %s, "mtp_fixed_depth": %s}}}\n' "$on" "$fixed" \
      > "$OMLX_BASE_PATH/model_settings.json"
    exec "$omlx/.venv/bin/omlx" serve --model-dir "$root" --port "$port" --host 127.0.0.1 --max-concurrent-requests 1 ;;
  ollama-*)
    # one model store per system, each holding a model named m (setup_ollama.sh); no cloud, one request at a time,
    # the model stays loaded between requests
    export OLLAMA_HOST="127.0.0.1:$port" OLLAMA_MODELS="$models/ollama-$sys" OLLAMA_NUM_PARALLEL=1 \
      OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_CONTEXT_LENGTH="$ctx" OLLAMA_KEEP_ALIVE=-1 OLLAMA_NO_CLOUD=1 OLLAMA_FLASH_ATTENTION=1
    exec "${OLLAMA:-ollama}" serve ;;
  *) echo "unknown system $sys" >&2; exit 2 ;;
esac
