#!/usr/bin/env bash
# Serve unsloth/DeepSeek-V4-Flash-0731-GGUF (DeepSeek-V4-Flash 0731, 284B-A38B
# MoE -- 43 layers, 256 routed + 1 shared expert, 6 experts/token) with the
# from-source llama.cpp server image on the DGX Spark (GB10 / sm_121a),
# accelerated with the model's own DSpark drafter for speculative decoding.
#
# WHY llama.cpp AND NOT vLLM FOR THIS ONE
# The FP8 source checkpoint is 155 GiB and does not fit this box. Of every
# published re-quantization, only three artifacts fit a 128 GB single-Spark:
# two GGUFs (this one and tekosML's GB10-tuned IQ2XXS, 80.8 GiB, measured
# 16.9 t/s target-only) and one vLLM artifact (rdtand/...-gridbook-87GB-spark-vllm,
# 81 GiB) that needs an out-of-tree `gridbook` pip plugin, trips a CUDA-graph
# bug on vLLM 0.27+, and ships **without** the DSpark sidecar. Every other
# vLLM-format quant (NVFP4, W4A16, GPTQ-Int4, AutoRound) is 142-170 GiB.
# Since DSpark is the single biggest decode lever on this hardware, and
# llama.cpp b10375 supports both the `deepseek4` arch and `--spec-type
# draft-dspark` natively with no plugin, the GGUF path wins outright.
#
# Two files from the one repo are involved:
#   - the target model    ${QUANT}/DeepSeek-V4-Flash-0731-${QUANT}-*.gguf  (~85 GB at IQ2_M, sharded)
#   - the DSpark drafter  dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf          (~10.1 GiB)
# Unsloth put the drafter at the repo ROOT precisely so one copy serves every
# quant folder and llama.cpp can auto-discover it. We still name it with -md
# anyway: an explicit path never depends on -hf having pulled the repo's
# auxiliary files, and it costs nothing when it has.
#
# DSpark is DeepSeek's own speculative module, extracted from the 0731
# checkpoint (25 FP8 projections + BF16 Markov/confidence heads + MXFP4 routed
# experts, arch `dflash`). It drafts several tokens per forward pass and the
# 284B verifies them in parallel -- identical output, ~1.9x decode. It needs
# --spec-type draft-dspark, not the plain draft-simple path.
#
# Requires a llama.cpp build >= b10269 (b10228 added DeepSeek-V4 DSpark, but
# b10259-b10268 advertise `draft-dspark` and then abort while loading the
# drafter -- avoid that window). build.lock pins b10375; ./build.sh --reproduce.
#
# Usage:
#   ./run-deepseek-v4-flash.sh                    # foreground (Ctrl-C to stop)
#   QUANT=UD-IQ1_M ./run-deepseek-v4-flash.sh     # smaller quant, more headroom
#   SPEC=0 ./run-deepseek-v4-flash.sh             # disable DSpark (frees ~10 GiB)
#   PARALLEL=1 CTX=262144 ./run-deepseek-v4-flash.sh  # back to the single-slot
#                                                 # config; fastest single stream
#   REASONING=high ./run-deepseek-v4-flash.sh     # none | high | max
#   CACHE_TYPE=f16 FLASH_ATTN=auto ./run-deepseek-v4-flash.sh   # fall back if a
#                                                 # future build rejects q8_0 KV
#   DETACH=1 ./run-deepseek-v4-flash.sh           # background server, restarts on boot
#   ./run-deepseek-v4-flash.sh --ctx-size 65536   # append/override any llama-server flag
#
# NOTE: weights + drafter are ~95 GiB of the Spark's shared 121 GiB. Nothing
# else substantial can be resident. Stop other GPU-heavy containers first --
# `docker ps` then `docker stop <name>`.
#
# Env: IMAGE, PORT (host), QUANT, CTX, PARALLEL, GPU_LAYERS, SPEC, DRAFT_MAX,
#      REASONING, FLASH_ATTN, CACHE_TYPE, HF_TOKEN, HF_HOME, DETACH.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-llama-spark:latest}"
REPO="unsloth/DeepSeek-V4-Flash-0731-GGUF"
QUANT="${QUANT:-UD-IQ2_M}"             # 84.68 GiB; + 10.15 GiB drafter = 94.8 GiB resident
PORT="${PORT:-8080}"
CTX="${CTX:-524288}"                   # TOTAL across slots -- see the --ctx-size note below.
                                       # With PARALLEL=2 this is 262144 per slot, i.e. each slot
                                       # gets exactly the budget the old single-slot config had.
                                       # Measured on the Spark (q8_0 KV + DSpark drafter resident):
                                       #   32768 ctx -> 97918 MiB     262144 ctx -> 98895 MiB
                                       # 8x the context costs only +977 MiB, because most of the
                                       # 43 blocks are sliding-window (attention.sliding_window=128,
                                       # see also attention.compress_ratios) and so cost a CONSTANT
                                       # amount regardless of ctx; only the full-attention minority
                                       # scales. Fits ~672 MiB constant + ~4.4 KiB/token, leaving
                                       # ~25 GiB free of the GB10's 124610 MiB.
                                       # Native max is 1048576 (yarn x16 over a 65536 base) and by
                                       # the same fit would land ~102 GiB -- also fits; prefill
                                       # time, not memory, is what makes big ctx expensive here.
PARALLEL="${PARALLEL:-2}"              # 2 concurrent slots. CTX is TOTAL and is divided by this,
                                       # so CTX was doubled in step to keep 262144 per slot.
                                       # Extra cost over 1 slot is small: the KV pool is sized by
                                       # CTX regardless of slot count, so only the per-stream
                                       # sliding-window caches (~672 MiB each) and the shared
                                       # drafter's per-seq KV multiply -- roughly +1.5-2 GiB.
                                       # Throughput is the real trade: see the batching note below.
GPU_LAYERS="${GPU_LAYERS:-999}"        # 999 = offload every layer (whole model on GPU)
REASONING="${REASONING:-none}"         # reasoning_effort: none | high | max
DRAFT_MAX="${DRAFT_MAX:-3}"            # draft tokens per verify step; unsloth's measured default
FLASH_ATTN="${FLASH_ATTN:-on}"         # measured: accepted on this arch (see below)
CACHE_TYPE="${CACHE_TYPE:-q8_0}"       # measured: half the KV bytes AND faster than f16
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

# --- resolve what is already in the shared hub cache ------------------------
# Same idea (and same cache) as run.sh's resolve_cached_gguf: if the GGUF is
# already sitting in $HF_HOME/hub, serve it in place with -m rather than making
# llama.cpp re-download 85 GiB into its own separate cache under
# $HF_HOME/llama.cpp. Echoes the host path of the first shard, or nothing.
# Sharded models only need the first shard named -- llama-server finds the rest
# in the same directory.
resolve_cached_gguf() {
  local repo="$1" pat="$2" cache_repo snap f first
  local -a ggufs
  cache_repo="$HF_HOME/hub/models--${repo//\//--}"
  [[ -d "$cache_repo/snapshots" ]] || return 0
  snap=""
  [[ -f "$cache_repo/refs/main" ]] && snap="$cache_repo/snapshots/$(<"$cache_repo/refs/main")"
  [[ -d "$snap" ]] || snap="$(find "$cache_repo/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
  [[ -d "$snap" ]] || return 0
  mapfile -t ggufs < <(find -L "$snap" -type f -iname "$pat" 2>/dev/null | sort)
  [[ ${#ggufs[@]} -eq 0 ]] && return 0
  first=""
  for f in "${ggufs[@]}"; do
    [[ "$f" == *-00001-of-* ]] && { first="$f"; break; }
  done
  printf '%s\n' "${first:-${ggufs[0]}}"
}

# HF_HOME is bind-mounted at /root/.cache/huggingface, so a host path under it
# maps by prefix swap. The hub cache stores snapshots as relative symlinks into
# blobs/, which resolve fine inside the container because the whole tree is
# mounted.
to_ctr() { printf '/root/.cache/huggingface/%s\n' "${1#"${HF_HOME}"/}"; }

TARGET_HOST="$(resolve_cached_gguf "$REPO" "*${QUANT}*.gguf")"

# --- DSpark drafter ---------------------------------------------------------
# Look in both caches before fetching: the hub cache (if it was pulled with
# `hf download`) and llama.cpp's own cache (if a previous -hf run pulled the
# repo's root-level auxiliary files alongside the quant). Only fetch by URL
# when neither has it. Kept outside the llama.cpp/ subdir when we do fetch it,
# because that one is created root-owned by the container and isn't writable
# from the host. Resumable, and skipped when the file is already complete.
DRAFT_NAME="dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
DRAFT_SIZE=10896057440
DRAFT_CTR_PATH=""

if [[ "${SPEC:-1}" == 1 ]]; then
  cached="$(resolve_cached_gguf "$REPO" "$DRAFT_NAME")"
  if [[ -z "$cached" ]]; then
    cached="$(ls -1 "${HF_HOME}"/llama.cpp/models--unsloth--DeepSeek-V4-Flash-0731-GGUF/snapshots/*/"${DRAFT_NAME}" 2>/dev/null | head -n1 || true)"
  fi
  if [[ -n "$cached" && "$(stat -Lc %s "$cached")" == "$DRAFT_SIZE" ]]; then
    DRAFT_CTR_PATH="$(to_ctr "$cached")"
  else
    DRAFT_HOST_PATH="${HF_HOME}/deepseek-v4-flash/${DRAFT_NAME}"
    DRAFT_CTR_PATH="$(to_ctr "$DRAFT_HOST_PATH")"
    mkdir -p "$(dirname "$DRAFT_HOST_PATH")"
    if [[ ! -f "$DRAFT_HOST_PATH" || "$(stat -c %s "$DRAFT_HOST_PATH")" != "$DRAFT_SIZE" ]]; then
      echo ">> fetching DSpark drafter ($((DRAFT_SIZE / 1024 / 1024)) MiB) ..."
      auth=()
      [[ -n "${HF_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer ${HF_TOKEN}")
      curl -fL --progress-bar -C - "${auth[@]+"${auth[@]}"}" \
        -o "$DRAFT_HOST_PATH" \
        "https://huggingface.co/${REPO}/resolve/main/${DRAFT_NAME}"
    fi
  fi
fi

# --- llama-server options --------------------------------------------------
# --ctx-size is LOAD-BEARING. deepseek4.context_length in the GGUF is 1048576,
# and llama.cpp's default (--ctx-size 0) means "take it from the model" -- which
# on this box tries to size a million-token KV pool on top of 95 GiB of weights
# and dies. Always pass an explicit budget. It is the TOTAL split across slots,
# so PARALLEL>1 divides it rather than multiplying: at PARALLEL=2 the default
# CTX=524288 gives each slot 262144. Raise CTX, not PARALLEL alone, or slots
# shrink under you -- context shift is OFF by default in this build, so a slot
# that fills stops mid-generation with truncated=1 rather than sliding.
#
# WHAT PARALLEL=2 ACTUALLY COSTS (unmeasured -- see the measured table below for
# the PARALLEL=1 baseline). Decode here is memory-bandwidth-bound, so batching
# 2 streams does NOT cost 2x: the dense/attention weights are read once per step
# and amortise across the batch. The MoE half does not amortise nearly as well
# -- 2 tokens each pick 6 of 256 routed experts, and with 256 experts to choose
# from the overlap is slight, so expert traffic nearly doubles. Expect aggregate
# throughput up but short of 2x, and per-stream t/s to fall, when both slots run.
#
# DSpark compounds this. Verification already batches DRAFT_MAX+1 = 4 tokens per
# slot; at PARALLEL=2 the step batch is up to 8. That is the same amortisation
# headroom batching wants, so the drafter's measured +54% shrinks as concurrency
# rises. If per-stream latency matters more than aggregate, lower DRAFT_MAX.
#
# Prefill, not decode, is the contention that bites: measured 335-350 t/s, so a
# 47k-token prompt is ~140 s during which the other slots' decode is throttled
# by the interleaved n_batch=2048 chunks.
#
# Slot selection is by prompt prefix similarity (-sps, default 0.10) and falls
# back to LRU, so sequential turns of one conversation keep landing on the slot
# holding their cache. Two unrelated conversations each get their own slot cache
# within their own 262144 budget.
#
# KV geometry: head_count_kv=1 with key/value_length 512 (MLA-style compressed
# latent) over 43 layers = ~44k elements/token, so ~86 KiB/token at f16 --
# ~2.8 GiB at 32768. q8_0 roughly halves that but llama.cpp needs flash
# attention for a quantized V cache, hence the CACHE_TYPE/FLASH_ATTN pairing.
#
# FLASH_ATTN=on + q8_0 KV is MEASURED, not assumed. The worry was that
# DeepSeek-V4 attention carries a sparse indexer (index_topk 512 over 64 index
# heads) alongside the compressed latent path and might reject flash attention
# on an arch this new. It does not: b10375 accepts -fa on with a q8_0 K/V cache
# and it is the fastest configuration measured here, so it is the default.
# Keep FLASH_ATTN=auto CACHE_TYPE=f16 as the fallback if a future build regresses.
#
# Measured on this box (image ba360efe / b10375, UD-IQ2_M, 300-token single
# stream, temperature 0, warm, 21-token prompt):
#
#   | config                          | decode    | draft accepted |
#   |---------------------------------|-----------|----------------|
#   | SPEC=0 (no drafter)             | 20.57 t/s | --             |
#   | SPEC=1, f16 KV, -fa auto        | 30.01 t/s | 180/354 (51%)  |
#   | SPEC=1, q8_0 KV, -fa on (deflt) | 31.58 t/s | 184/343 (54%)  |
#
# DSpark is worth +54% decode over no drafter. Model load is ~80-100 s.
# Resident is ~102 GiB of the 121 GiB either way -- weights dominate, so the
# q8_0 KV saving buys context headroom rather than a smaller footprint.
#
# Sampling follows DeepSeek's own published eval settings for 0731
# (temperature 1.0, top_p 0.95); the GGUF's baked-in metadata says top_p 1.0.
# These are server-side defaults -- clients still override per request.
#
# --jinja applies the template Unsloth embedded in the GGUF. It matters more
# than usual here: the base checkpoint ships NO chat template at all, only a
# programmatic encoder (encoding/encoding_dsv4.py), so this Jinja port is the
# only thing that produces the right control tokens, and tool calling depends
# on it. Tool calls come back in DeepSeek's DSML block form, which the build's
# DeepSeek V3.2 parser handles.
#
# REASONING defaults to `none`, which injects nothing. That is the template's
# own default and it is deliberate: `high` and `max` prepend maximum-
# deliberation system prompts ("leave absolutely nothing to chance...") that
# can run a trivial prompt for many minutes. This is the same trap vLLM's
# deepseek_v4 path falls into by defaulting to high -- here we simply don't.
server_args=(
  -ngl "${GPU_LAYERS}"
  --ctx-size "${CTX}"
  --flash-attn "${FLASH_ATTN}"
  --cache-type-k "${CACHE_TYPE}"
  --cache-type-v "${CACHE_TYPE}"
  --parallel "${PARALLEL}"
  --jinja
  --temp 1.0
  --top-p 0.95
)

if [[ -n "$TARGET_HOST" ]]; then
  server_args=(-m "$(to_ctr "$TARGET_HOST")" "${server_args[@]}")
else
  server_args=(-hf "${REPO}:${QUANT}" "${server_args[@]}")
fi

[[ "${REASONING}" == "none" ]] || \
  server_args+=(--chat-template-kwargs "{\"reasoning_effort\":\"${REASONING}\"}")

if [[ "${SPEC:-1}" == 1 ]]; then
  server_args+=(
    -md "$DRAFT_CTR_PATH"
    --spec-type draft-dspark
    -ngld 999
    --spec-draft-n-max "${DRAFT_MAX}"
  )
fi

# --- docker run ------------------------------------------------------------
# Keep models in the standard Hugging Face cache (HF_HOME, default
# ~/.cache/huggingface), shared with any other HF tooling on the host. The GGUF
# shards download via -hf into llama.cpp's cache under it; reused on later
# runs. The first run pulls ~85 GiB, so expect a long wait before the server
# binds.
run_flags=(--gpus all --ipc=host -p "${PORT}:8080"
           -e "HF_TOKEN=${HF_TOKEN:-}"
           -v "${HF_HOME}:/root/.cache/huggingface"
           -e "HF_HOME=/root/.cache/huggingface"
           -e "LLAMA_CACHE=/root/.cache/huggingface/llama.cpp")

if [[ "${DETACH:-0}" == 1 ]]; then
  run_flags+=(-d --name deepseek-v4-flash --restart unless-stopped)
else
  run_flags+=(--rm -it)
fi

echo ">> serving ${REPO}:${QUANT}${TARGET_HOST:+ (cached)}  (spec=${SPEC:-1} reasoning=${REASONING} kv=${CACHE_TYPE} fa=${FLASH_ATTN})"
echo ">> ${PARALLEL} slot(s) sharing ${CTX} ctx"
echo ">> http://localhost:${PORT}  (OpenAI-compatible: /v1/chat/completions , Web UI at /)"
set -x
exec docker run "${run_flags[@]}" "$IMAGE" "${server_args[@]}" "$@"
