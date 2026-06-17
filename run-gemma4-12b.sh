#!/usr/bin/env bash
# Serve unsloth/gemma-4-12b-it-GGUF (a heavily-quantized Dynamic 2.0 build, by
# default UD-Q4_K_XL) with the from-source llama.cpp server image on the DGX
# Spark (GB10 / sm_121a). Dense Gemma 4 12B is a plain autoregressive model, so
# this is a normal `llama-server` HTTP endpoint -- styled like the sibling
# vLLM run-qwen3.6.sh: model pinned here, extra flags pass through.
#
# Usage:
#   ./run-gemma4-12b.sh                    # foreground (Ctrl-C to stop)
#   QUANT=UD-Q5_K_XL ./run-gemma4-12b.sh   # pick a different quant
#   DETACH=1 ./run-gemma4-12b.sh           # background server, restarts on boot
#   ./run-gemma4-12b.sh --ctx-size 131072  # append/override any llama-server flag
#
# Env: IMAGE, PORT (host), QUANT, GPU_LAYERS, HF_TOKEN, HF_HOME.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-llama-spark:latest}"
REPO="unsloth/gemma-4-12b-it-GGUF"
QUANT="${QUANT:-UD-Q4_K_XL}"           # "really restricted" Dynamic 2.0 4-bit
PORT="${PORT:-8080}"
GPU_LAYERS="${GPU_LAYERS:-999}"        # 999 = offload every layer (whole model on GPU)
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

# --- llama-server options --------------------------------------------------
# Sampling follows Unsloth's recommended Gemma settings (temp 1.0, top_k 64,
# top_p 0.95, min_p 0.0). These are server defaults; clients can still override
# per request. -fa on + q8_0 KV cache keeps a long context cheap on the Spark's
# unified memory. --jinja (on by default) uses the model's own chat template.
server_args=(
  -hf "${REPO}:${QUANT}"
  -ngl "${GPU_LAYERS}"
  --ctx-size 65536
  --flash-attn on
  --cache-type-k q8_0
  --cache-type-v q8_0
  --parallel 1
  --jinja
  --temp 1.0
  --top-k 64
  --top-p 0.95
  --min-p 0.0
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
  run_flags+=(-d --name gemma4-12b --restart unless-stopped)
else
  run_flags+=(--rm -it)
fi

echo ">> serving ${REPO}:${QUANT}"
echo ">> http://localhost:${PORT}  (OpenAI-compatible: /v1/chat/completions , Web UI at /)"
set -x
exec docker run "${run_flags[@]}" "$IMAGE" "${server_args[@]}" "$@"
