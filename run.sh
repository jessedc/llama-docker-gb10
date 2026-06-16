#!/usr/bin/env bash
# Serve a model with the locally built llama.cpp server image on the DGX Spark.
#
# Usage:
#   ./run.sh ggml-org/gemma-3-4b-it-GGUF            # pull + serve a HF GGUF repo
#   ./run.sh ggml-org/gemma-3-4b-it-GGUF:Q8_0       # pin a specific quant
#   ./run.sh /abs/path/to/model.gguf               # serve a local GGUF file
#   ./run.sh <model> --ctx-size 32768 --parallel 4 # extra flags pass to llama-server
#
# Env:
#   IMAGE          image to run             (default: llama-spark:latest)
#   PORT           host port               (default: 8080)
#   GPU_LAYERS     -ngl / layers on GPU    (default: 999 = all)
#   HF_TOKEN       Hugging Face token for gated/private repos
#   LLAMA_CACHE    host cache for -hf downloads (default: ~/.cache/llama.cpp)
#   MODELS_DIR     host dir for local -m models (default: dirname of the path arg)
#   DETACH=1       run detached + restart (server mode) instead of interactive
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-llama-spark:latest}"
PORT="${PORT:-8080}"
GPU_LAYERS="${GPU_LAYERS:-999}"
LLAMA_CACHE="${LLAMA_CACHE:-$HOME/.cache/llama.cpp}"

MODEL="${1:-}"; shift || true
if [[ -z "$MODEL" ]]; then
  sed -n '2,11p' "$0"; exit 2
fi

run_flags=(--gpus all --ipc=host -p "${PORT}:8080"
           -e "HF_TOKEN=${HF_TOKEN:-}"
           -e "GPU_LAYERS=${GPU_LAYERS}")

# llama-server args. Host/port come from the image's LLAMA_ARG_* env defaults.
server_args=(-ngl "${GPU_LAYERS}")

if [[ -e "$MODEL" ]]; then
  # Local GGUF: mount its directory read-only and serve by path.
  abs="$(readlink -f "$MODEL")"
  dir="$(dirname "$abs")"
  base="$(basename "$abs")"
  run_flags+=(-v "${dir}:/models:ro")
  server_args+=(-m "/models/${base}")
  echo ">> serving local model /models/${base}"
else
  # Treat as a Hugging Face repo (optionally repo:quant); cache downloads on host.
  mkdir -p "$LLAMA_CACHE"
  run_flags+=(-v "${LLAMA_CACHE}:/root/.cache/llama.cpp")
  server_args+=(-hf "$MODEL")
  echo ">> serving HF model $MODEL (cache: $LLAMA_CACHE)"
fi

if [[ "${DETACH:-0}" == 1 ]]; then
  run_flags+=(-d --name llama --restart unless-stopped)
else
  run_flags+=(--rm -it)
fi

echo ">> http://localhost:${PORT}  (OpenAI-compatible: /v1/chat/completions , Web UI at /)"
set -x
exec docker run "${run_flags[@]}" "$IMAGE" "${server_args[@]}" "$@"
