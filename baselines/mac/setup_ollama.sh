#!/usr/bin/env bash
# bash baselines/mac/setup_ollama.sh SYSTEM: make the model "m" of an Ollama system (serve.sh) in its own store
# (MODELS/ollama-SYSTEM). Starts a private server on PORT (default 18601), creates, stops. Copies the weights (17-19 GB).
set -euo pipefail
sys="${1:?system}"; port="${PORT:-18601}"; models="${MODELS:-$HOME/models}"
work="$models/ollama-src/$sys"; mkdir -p "$work"
case "$sys" in
  ollama-q4)  printf 'FROM %s\n' "$models/gguf-bartowski/Inkling-2-Mini-Q4_K_M.gguf" > "$work/Modelfile" ;;
  ollama-mlx) printf 'FROM %s\n' "$models/Inkling-2-Mini" > "$work/Modelfile" ;;
  ollama-mlx-ds)
    # DRAFT must be a local safetensors directory other than FROM: the DSpark head and the config that describes it
    d="$work/draft"; mkdir -p "$d"; for f in mtp.safetensors config.json; do ln -sfn "$models/Inkling-2-Mini/$f" "$d/$f"; done
    printf 'FROM %s\nDRAFT %s\n' "$models/Inkling-2-Mini" "$d" > "$work/Modelfile" ;;
  *) echo "unknown Ollama system $sys" >&2; exit 2 ;;
esac
printf 'PARAMETER num_ctx 8192\n' >> "$work/Modelfile"
export OLLAMA_HOST="127.0.0.1:$port" OLLAMA_MODELS="$models/ollama-$sys" OLLAMA_NO_CLOUD=1
ollama serve > "$work/serve.log" 2>&1 & pid=$!; trap 'kill $pid 2>/dev/null' EXIT
for _ in $(seq 1 60); do curl -sf "http://$OLLAMA_HOST/api/version" > /dev/null && break; sleep 1; done
cd "$work" && ollama create m -f Modelfile ${CREATE_FLAGS:-}
ollama show m
