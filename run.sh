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
#   HF_HOME        shared host model cache (default: ~/.cache/huggingface,
#                  same dir/default as the sibling vLLM setup -> one common
#                  model store). llama.cpp's -hf cache lands in a llama.cpp/
#                  subdir of it (its flat layout differs from the HF hub layout).
#   DETACH=1       run detached + restart (server mode) instead of interactive
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-llama-spark:latest}"
PORT="${PORT:-8080}"
GPU_LAYERS="${GPU_LAYERS:-999}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Look for an already-downloaded GGUF for an HF repo[:quant] in the shared HF
# hub cache, so we can serve it in place (-m) instead of re-downloading via -hf
# into llama.cpp's separate flat cache. Echoes the host path of the GGUF (the
# first shard, for sharded models) on success; prints nothing if none is found.
resolve_cached_gguf() {
  local spec="$1" repo quant cache_repo snap f first
  repo="${spec%%:*}"
  quant=""
  [[ "$spec" == *:* ]] && quant="${spec#*:}"
  cache_repo="$HF_HOME/hub/models--${repo//\//--}"
  [[ -d "$cache_repo/snapshots" ]] || return 0
  # Prefer the commit refs/main points at; else the newest snapshot dir.
  snap=""
  [[ -f "$cache_repo/refs/main" ]] && snap="$cache_repo/snapshots/$(<"$cache_repo/refs/main")"
  [[ -d "$snap" ]] || snap="$(find "$cache_repo/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
  [[ -d "$snap" ]] || return 0
  # Candidate GGUFs (follow symlinks into blobs/), optionally filtered by quant.
  local -a ggufs
  mapfile -t ggufs < <(find -L "$snap" -type f -iname "*${quant}*.gguf" 2>/dev/null | sort)
  [[ ${#ggufs[@]} -eq 0 ]] && return 0
  # For sharded models serve the first shard; llama-server loads the rest.
  first=""
  for f in "${ggufs[@]}"; do
    [[ "$f" == *-00001-of-* ]] && { first="$f"; break; }
  done
  printf '%s\n' "${first:-${ggufs[0]}}"
}

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
  # Treat as a Hugging Face repo (optionally repo:quant). Share one host model
  # store with vLLM (HF_HOME, default ~/.cache/huggingface).
  mkdir -p "$HF_HOME"
  cached_gguf="$(resolve_cached_gguf "$MODEL")"
  if [[ -n "$cached_gguf" ]]; then
    # A GGUF for this repo is already in the shared HF hub cache (e.g. pulled by
    # `hf download`): serve it in place by path, no second download/copy.
    run_flags+=(-v "${HF_HOME}:/root/.cache/huggingface:ro")
    server_args+=(-m "/root/.cache/huggingface/${cached_gguf#"$HF_HOME"/}")
    echo ">> serving cached GGUF $MODEL"
    echo "   reused in place: $cached_gguf"
  else
    # Not cached as GGUF: download via -hf into the shared store; llama.cpp's
    # own flat cache lands in a llama.cpp/ subdir of it.
    run_flags+=(-v "${HF_HOME}:/root/.cache/huggingface"
                -e "HF_HOME=/root/.cache/huggingface"
                -e "LLAMA_CACHE=/root/.cache/huggingface/llama.cpp")
    server_args+=(-hf "$MODEL")
    echo ">> serving HF model $MODEL (shared cache: $HF_HOME)"
  fi
fi

if [[ "${DETACH:-0}" == 1 ]]; then
  run_flags+=(-d --name llama --restart unless-stopped)
else
  run_flags+=(--rm -it)
fi

echo ">> http://localhost:${PORT}  (OpenAI-compatible: /v1/chat/completions , Web UI at /)"
set -x
exec docker run "${run_flags[@]}" "$IMAGE" "${server_args[@]}" "$@"
