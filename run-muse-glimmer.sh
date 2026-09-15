#!/usr/bin/env bash
# Serve unsloth/Muse-Glimmer-30B-GGUF (Meta's Muse Glimmer 30B, by default the
# UD-Q6_K_XL Dynamic 2.0 quant) with the from-source llama.cpp server image on
# the DGX Spark (GB10 / sm_121a) -- multimodal, and accelerated with the DFlash
# drafter for speculative decoding.
#
# Three files from the one repo are involved:
#   - the text model          Muse-Glimmer-30B-${QUANT}.gguf   (~25 GB at Q6)
#   - the perception encoder  mmproj-...-Q8_0.gguf             (~2.0 GB, vision)
#   - the DFlash drafter      dflash-kquant.gguf               (~1.6 GB)
# -hf pulls all three (it fetches the repo's auxiliary files alongside the
# model; --no-mmproj opts out of the encoder). But the drafter still has to be
# named explicitly with -md, and it cannot be named via -hfd/--hf-repo-draft:
# "dflash-kquant" is not a valid HF quantization tag, so that endpoint 400s and
# the lookup would fall back to the wrong file. So we point -md at the copy in
# llama.cpp's cache, and fetch it by URL ourselves only if it isn't there yet.
#
# DFlash is a block-diffusion drafter: it proposes a whole block of tokens per
# forward pass and the 30B verifies them in parallel, so decode gets faster
# with identical output. It needs --spec-type draft-dflash -- not the plain
# draft-simple path used by ordinary same-family speculative decoding.
#
# Requires a llama.cpp build >= b10353 (Muse Glimmer support, upstream #26841).
# build.lock pins b10375; rebuild with ./build.sh --reproduce.
#
# Usage:
#   ./run-muse-glimmer.sh                    # foreground (Ctrl-C to stop)
#   QUANT=UD-Q4_K_XL ./run-muse-glimmer.sh   # pick a different quant
#   PARALLEL=1 ./run-muse-glimmer.sh         # single slot (default is 4)
#   SPEC=0 ./run-muse-glimmer.sh             # disable speculative decoding
#   MMPROJ=0 ./run-muse-glimmer.sh           # text-only (skips the 3.8 GB encoder)
#   REASONING=high ./run-muse-glimmer.sh     # low | medium | high | xhigh
#   DETACH=1 ./run-muse-glimmer.sh           # background server, restarts on boot
#   ./run-muse-glimmer.sh --ctx-size 262144  # append/override any llama-server flag
#
# NOTE: the Spark's 121 GB of unified memory is shared with anything else on
# the box. Stop other GPU-heavy containers first -- `docker ps` then
# `docker stop <name>` -- or this won't fit.
#
# Env: IMAGE, PORT (host), QUANT, CTX (per slot), PARALLEL, GPU_LAYERS, SPEC,
#      DRAFT_MAX, MMPROJ, REASONING, HF_TOKEN, HF_HOME, DETACH.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-llama-spark:latest}"
REPO="unsloth/Muse-Glimmer-30B-GGUF"
QUANT="${QUANT:-UD-Q6_K_XL}"           # ~26 GB; near-lossless Dynamic 2.0 6-bit
PORT="${PORT:-8080}"
CTX="${CTX:-131072}"                   # PER SLOT; native is 131072, model tops out at 262144
PARALLEL="${PARALLEL:-4}"              # concurrent request slots; idle slots cost nothing
GPU_LAYERS="${GPU_LAYERS:-999}"        # 999 = offload every layer (whole model on GPU)
REASONING="${REASONING:-low}"          # reasoning_strength: low|medium|high|xhigh
DRAFT_MAX="${DRAFT_MAX:-4}"            # draft tokens per verify step (upstream default 3)
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

# --- DFlash drafter: fetch once into the shared cache ------------------------
# Kept outside the llama.cpp/ subdir because that one is created root-owned by
# the container and isn't writable from the host. Resumable, and skipped when
# the file is already complete.
DRAFT_NAME="dflash-kquant.gguf"
DRAFT_SIZE=1631205312
DRAFT_CTR_PATH=""

if [[ "${SPEC:-1}" == 1 ]]; then
  # In practice -hf pulls the repo's auxiliary files too, so after one run the
  # drafter is already sitting in llama.cpp's cache under a snapshot hash.
  # Prefer that copy; only fetch our own when it isn't there (first run, or a
  # llama.cpp that stops doing this), which keeps us off a second 1.6 GB pull.
  cached="$(ls -1 "${HF_HOME}"/llama.cpp/models--unsloth--Muse-Glimmer-30B-GGUF/snapshots/*/"${DRAFT_NAME}" 2>/dev/null | head -n1 || true)"
  if [[ -n "$cached" && "$(stat -Lc %s "$cached")" == "$DRAFT_SIZE" ]]; then
    DRAFT_CTR_PATH="/root/.cache/huggingface/${cached#"${HF_HOME}"/}"
  else
    DRAFT_HOST_PATH="${HF_HOME}/muse-glimmer/${DRAFT_NAME}"
    DRAFT_CTR_PATH="/root/.cache/huggingface/muse-glimmer/${DRAFT_NAME}"
    mkdir -p "$(dirname "$DRAFT_HOST_PATH")"
    if [[ ! -f "$DRAFT_HOST_PATH" || "$(stat -c %s "$DRAFT_HOST_PATH")" != "$DRAFT_SIZE" ]]; then
      echo ">> fetching DFlash drafter ($((DRAFT_SIZE / 1024 / 1024)) MiB) ..."
      auth=()
      [[ -n "${HF_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer ${HF_TOKEN}")
      curl -fL --progress-bar -C - "${auth[@]+"${auth[@]}"}" \
        -o "$DRAFT_HOST_PATH" \
        "https://huggingface.co/${REPO}/resolve/main/${DRAFT_NAME}"
    fi
  fi
fi

# --- llama-server options --------------------------------------------------
# Sampling follows Meta's / Unsloth's recommended Muse Glimmer settings
# (temp 1.0, top_k 64, top_p 0.95). These are server defaults; clients can
# still override per request. -fa on + q8_0 KV cache keeps the 131k context
# affordable on the Spark's unified memory. --jinja applies the model's own
# control-token chat template, which tool calling depends on.
# --ctx-size is the TOTAL budget llama-server splits across slots, so raising
# --parallel against a fixed total silently shrinks every slot (4 slots against
# 131072 leaves each request 32768). CTX here is therefore PER SLOT and the
# total is multiplied out, which keeps `PARALLEL=4` honest: 4 x 131072.
server_args=(
  -hf "${REPO}:${QUANT}"
  -ngl "${GPU_LAYERS}"
  --ctx-size "$(( CTX * PARALLEL ))"
  --flash-attn on
  --cache-type-k q8_0
  --cache-type-v q8_0
  --parallel "${PARALLEL}"
  --jinja
  --chat-template-kwargs "{\"reasoning_strength\":\"${REASONING}\"}"
  --temp 1.0
  --top-k 64
  --top-p 0.95
)

[[ "${MMPROJ:-1}" == 1 ]] || server_args+=(--no-mmproj)

if [[ "${SPEC:-1}" == 1 ]]; then
  server_args+=(
    -md "$DRAFT_CTR_PATH"
    --spec-type draft-dflash
    -ngld 999
    --spec-draft-n-max "${DRAFT_MAX}"
  )
fi

# --- docker run ------------------------------------------------------------
# Keep models in the standard Hugging Face cache (HF_HOME, default
# ~/.cache/huggingface), shared with any other HF tooling on the host. The GGUFs
# download via -hf into llama.cpp's cache under it; reused on later runs.
run_flags=(--gpus all --ipc=host -p "${PORT}:8080"
           -e "HF_TOKEN=${HF_TOKEN:-}"
           -v "${HF_HOME}:/root/.cache/huggingface"
           -e "HF_HOME=/root/.cache/huggingface"
           -e "LLAMA_CACHE=/root/.cache/huggingface/llama.cpp")

if [[ "${DETACH:-0}" == 1 ]]; then
  run_flags+=(-d --name muse-glimmer --restart unless-stopped)
else
  run_flags+=(--rm -it)
fi

echo ">> serving ${REPO}:${QUANT}  (vision=${MMPROJ:-1} spec=${SPEC:-1} reasoning=${REASONING})"
echo ">> ${PARALLEL} slot(s) x ${CTX} ctx = $(( CTX * PARALLEL )) total"
echo ">> http://localhost:${PORT}  (OpenAI-compatible: /v1/chat/completions , Web UI at /)"
set -x
exec docker run "${run_flags[@]}" "$IMAGE" "${server_args[@]}" "$@"
