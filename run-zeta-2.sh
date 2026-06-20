#!/usr/bin/env bash
# Serve bartowski/zed-industries_zeta-2-GGUF (Zed's Zeta-2 edit-prediction model,
# by default the Q8_0 quant) with the from-source llama.cpp server image on the
# DGX Spark (GB10 / sm_121a). A normal `llama-server` HTTP endpoint -- styled
# like the sibling run-gemma4-12b.sh: model pinned here, extra flags pass through.
#
# Usage:
#   ./run-zeta-2.sh                     # foreground (Ctrl-C to stop)
#   QUANT=Q6_K ./run-zeta-2.sh          # pick a different quant
#   DETACH=1 ./run-zeta-2.sh            # background server, restarts on boot
#   ./run-zeta-2.sh --ctx-size 131072   # append/override any llama-server flag
#
# Env: IMAGE, PORT (host), QUANT, GPU_LAYERS, HF_TOKEN, HF_HOME.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-llama-spark:latest}"
REPO="bartowski/zed-industries_zeta-2-GGUF"
QUANT="${QUANT:-Q8_0}"
PORT="${PORT:-8080}"
GPU_LAYERS="${GPU_LAYERS:-999}"        # 999 = offload every layer (whole model on GPU)
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

# --- llama-server options --------------------------------------------------
# -fa on + q8_0 KV cache keeps a long context cheap on the Spark's unified
# memory. --jinja (on by default) uses the model's own chat template.
server_args=(
  -hf "${REPO}:${QUANT}"
  -ngl "${GPU_LAYERS}"
  --ctx-size 65536
  --flash-attn on
  --cache-type-k q8_0
  --cache-type-v q8_0
  --parallel 1
  --jinja
)

# --- docker run ------------------------------------------------------------
# Share one host model store with vLLM/llama.cpp (HF_HOME). The GGUF downloads
# via -hf into llama.cpp's flat cache under it; reused on later runs.
run_flags=(--gpus all --ipc=host -p "${PORT}:8080"
           -e "HF_TOKEN=${HF_TOKEN:-}"
           -v "${HF_HOME}:/root/.cache/huggingface"
           -e "HF_HOME=/root/.cache/huggingface"
           -e "LLAMA_CACHE=/root/.cache/huggingface/llama.cpp")

if [[ "${DETACH:-0}" == 1 ]]; then
  run_flags+=(-d --name zeta-2 --restart unless-stopped)
else
  run_flags+=(--rm -it)
fi

echo ">> serving ${REPO}:${QUANT}"
echo ">> http://localhost:${PORT}  (OpenAI-compatible: /v1/chat/completions , Web UI at /)"
set -x
exec docker run "${run_flags[@]}" "$IMAGE" "${server_args[@]}" "$@"
