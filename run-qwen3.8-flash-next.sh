#!/usr/bin/env bash
# Serve unsloth/Qwen3.8-Flash-Next-GGUF (Qwen3.8-Flash-Next -- 125B MoE, arch
# `qwen4exp`, 48 layers, 512 routed experts / 10 per token, multimodal, 262144
# native ctx) with the from-source llama.cpp server image on the DGX Spark
# (GB10 / sm_121a), with vision, accelerated by the model's own MTP head for
# speculative decoding.
#
# Three files from the one repo are involved:
#   - the target model    UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-*.gguf  (87.24 GiB, 3 shards)
#   - the MTP head        MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf  (1.78 GiB)
#   - the vision encoder  mmproj-F16.gguf                                (0.84 GiB)
#
# `hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-IQ4_XS/*" \
#    --include "MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf" --include "mmproj-F16.gguf"`
# pre-stages all three into $HF_HOME/hub and the script serves them in place.
#
# WHY UD-IQ4_XS
# Quants of this model are big for their bit width: the Ngram / per-layer-
# embedding (PLE) lookup table is kept at >= 4-bit because it is read at random
# and heavy quantization damages it. UD-IQ4_XS (KLD 0.084, top-1 89.6%) is the
# largest quant that leaves real headroom here. The next step up, UD-Q4_K_XL
# (KLD 0.047, top-1 92.3%), is 103.7 GiB before KV, the head or the desktop.
#
# THE MTP HEAD
# The model's own multi-token-prediction layer, extracted to a separate GGUF.
# It drafts the next token(s) and the 125B verifies them in one pass: identical
# output, faster decode. `shared-` heads borrow the token embedding and output
# projection from the loaded target instead of carrying copies (0.82 GiB less
# than the self-contained Q4_K_M; drafts identically). Q4_K_M rather than
# Unsloth's recommended shared-Q8_0 trades ~2 points of acceptance (their B200
# numbers: 64.4% vs 66.1%) for another 0.82 GiB of headroom.
#
# -md is ALWAYS passed explicitly. The head lives in the repo's MTP/ subfolder,
# which sidecar auto-discovery does not search, so `--spec-type draft-mtp` on
# its own finds nothing and silently runs at target-only speed. So the head is
# resolved to a path here, the same way run-deepseek-v4-flash.sh resolves its
# DSpark drafter.
#
# This build logs these two lines at startup. They are EXPECTED:
#   E llama_init_from_model: failed to initialize the context: qwen4exp requires ctx_other to be set (this warning is normal during memory fitting)
#   W operator(): failed to measure the memory of the extra model, fitting without it: ...
# (Unsloth's MTP README shows a `borrow_shared_tensor: this model is a draft
# head` variant; same cause.) The auto-fit tries to size the head on its own,
# before the target exists, so there is nothing for it to borrow from.
# Speculation still runs. Confirm it with `draft acceptance = ...` in
# `docker logs`. The one consequence is that the fit leaves out the head's
# memory, which is why -c and -ngl are always passed explicitly here.
#
# REQUIRES AN UNMERGED llama.cpp BUILD. Mainline has no MTP graph for `qwen4exp`
# and no cross-model tensor borrowing, so a stock build cannot use the head.
# build.lock pins d1a92352, the head of ggml-org/llama.cpp#28243 ("models:
# Qwen3.8-Flash-Next MTP"). Move to a release tag once that PR lands.
# ./build.sh --reproduce.
#
# Usage:
#   ./run-qwen3.8-flash-next.sh                    # foreground (Ctrl-C to stop)
#   SPEC=0 ./run-qwen3.8-flash-next.sh             # no MTP head (~-3 GiB, 1.3-1.6x slower)
#   MMPROJ=0 ./run-qwen3.8-flash-next.sh           # text-only (-1.1 GiB)
#   PARALLEL=1 CTX=262144 ./run-qwen3.8-flash-next.sh  # single slot, -5.5 GiB
#   REASONING=none ./run-qwen3.8-flash-next.sh     # none | low | medium | xhigh | default
#   DRAFT_MAX=3 ./run-qwen3.8-flash-next.sh        # draft tokens per verify step
#   DETACH=1 ./run-qwen3.8-flash-next.sh           # background server, restarts on boot
#   ./run-qwen3.8-flash-next.sh --ctx-size 65536   # append/override any llama-server flag
#
# NOTE: this config holds ~76 GiB of GPU buffers plus the 26.8 GiB PLE table in
# page cache, ~104 GiB of the Spark's shared 121 GiB. Nothing else substantial
# can be resident, so stop other GPU-heavy containers first (`docker ps`, then
# `docker stop <name>`).
#
# Env: IMAGE, PORT (host), QUANT, CTX, PARALLEL, GPU_LAYERS, SPEC, DRAFT_MAX,
#      MMPROJ, REASONING, FLASH_ATTN, CACHE_TYPE, HF_TOKEN, HF_HOME, DETACH.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-llama-spark:latest}"
REPO="unsloth/Qwen3.8-Flash-Next-GGUF"
QUANT="${QUANT:-UD-IQ4_XS}"            # 87.24 GiB (61.2 GiB on GPU + 26.8 GiB PLE mmap)
PORT="${PORT:-8080}"
CTX="${CTX:-524288}"                   # TOTAL across slots: 2 x 262144 (native max per slot).
                                       # Measured, 1 slot, q8_0 KV, head resident:
                                       #   32768 ctx -> 64804 MiB GPU    262144 ctx -> 70874 MiB GPU
                                       # 8x the context costs ~5.9 GiB: 4.4 GiB of KV (two 12-layer
                                       # caches, ~17 KiB/token at q8_0) plus a compute buffer that
                                       # grows 351 -> 1821 MiB. The other layers are recurrent, with
                                       # a fixed 450 MiB state per slot whatever the ctx.
PARALLEL="${PARALLEL:-2}"              # 2 slots, each keeping the full 262144 (CTX doubled in step).
                                       # The second 262144 costs +5.5 GiB GPU (71984 -> 77480 MiB,
                                       # both with vision + head). An idle slot costs no speed
                                       # (single-stream greedy 45.6 t/s here vs 46.0 at PARALLEL=1).
GPU_LAYERS="${GPU_LAYERS:-999}"        # 999 = offload every layer (whole model on GPU)
REASONING="${REASONING:-default}"      # default (template: xhigh) | medium | low | none
DRAFT_MAX="${DRAFT_MAX:-2}"            # measured best on sampled thinking output (see table)
MMPROJ="${MMPROJ:-1}"                  # vision encoder: +1.1 GiB, MTP still active with it
FLASH_ATTN="${FLASH_ATTN:-on}"
CACHE_TYPE="${CACHE_TYPE:-q8_0}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

# --- resolve what is already in the shared hub cache ------------------------
# Same logic (and cache) as run.sh / run-deepseek-v4-flash.sh: if a file is
# already in $HF_HOME/hub, serve it in place with -m rather than re-downloading
# ~87 GiB into llama.cpp's separate cache under $HF_HOME/llama.cpp. Echoes the
# host path (first shard, for sharded models), or nothing.
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
# maps by prefix swap (hub snapshots are relative symlinks into blobs/, which
# resolve inside the container because the whole tree is mounted).
to_ctr() { printf '/root/.cache/huggingface/%s\n' "${1#"${HF_HOME}"/}"; }

# Resolve an auxiliary file (repo-relative path + exact size) from the hub
# cache, then llama.cpp's own cache, and only fetch it by URL when neither has a
# complete copy. Fetched copies go under $HF_HOME/qwen3.8-flash-next/, not the
# llama.cpp/ subdir, which the container creates root-owned. Resumable, and
# skipped when already complete. Echoes the container path.
resolve_aux() {
  local rel="$1" size="$2" cached host
  local -a auth=()
  cached="$(resolve_cached_gguf "$REPO" "$(basename "$rel")")"
  if [[ -z "$cached" ]]; then
    cached="$(ls -1 "${HF_HOME}"/llama.cpp/models--"${REPO//\//--}"/snapshots/*/"${rel}" 2>/dev/null | head -n1 || true)"
  fi
  if [[ -n "$cached" && "$(stat -Lc %s "$cached")" == "$size" ]]; then
    to_ctr "$cached"
    return 0
  fi
  host="${HF_HOME}/qwen3.8-flash-next/${rel}"
  mkdir -p "$(dirname "$host")"
  if [[ ! -f "$host" || "$(stat -c %s "$host")" != "$size" ]]; then
    echo ">> fetching ${rel} ($((size / 1024 / 1024)) MiB) ..." >&2
    [[ -n "${HF_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer ${HF_TOKEN}")
    curl -fL --progress-bar -C - "${auth[@]+"${auth[@]}"}" \
      -o "$host" "https://huggingface.co/${REPO}/resolve/main/${rel}" >&2
  fi
  to_ctr "$host"
}

TARGET_HOST="$(resolve_cached_gguf "$REPO" "*${QUANT}*.gguf")"

DRAFT_CTR_PATH=""
if [[ "${SPEC:-1}" == 1 ]]; then
  DRAFT_CTR_PATH="$(resolve_aux MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf 1907151936)"
fi

MMPROJ_CTR_PATH=""
if [[ "$MMPROJ" == 1 ]]; then
  MMPROJ_CTR_PATH="$(resolve_aux mmproj-F16.gguf 904004000)"
fi

# --- llama-server options --------------------------------------------------
# --ctx-size is TOTAL across slots, so PARALLEL>1 divides it. Raise CTX along
# with PARALLEL or every slot shrinks. Always pass it: auto-fit does not count
# the MTP head (see the startup-log note above).
#
# MEMORY. The 26.8 GiB per_layer_token_embd (PLE) tensor is NOT copied to the
# GPU. llama.cpp keeps it CPU_Mapped with lazy reads, so it lives in the page
# cache and shows up in neither nvidia-smi nor free's "used". Measured at these
# defaults: 77480 MiB GPU at boot, and free "used" rose from 84.2 to 90.4 GiB
# over a benchmark session, most likely as the host-RAM prompt cache filled
# (its default --cache-ram cap is 8192 MiB). Budget ~76 GiB GPU + 26.8 GiB PLE + up to 8 GiB
# prompt cache. Squeeze it and the kernel evicts PLE pages to SSD. That should
# still work, but slower (unmeasured).
#
# Measured on this box (image d1a92352, UD-IQ4_XS, q8_0 KV, -fa on, warm,
# single stream). "greedy" = temp 0, no thinking, 400 tokens, mean of 2 warm
# runs on a code/explanation prompt. "think" = the server defaults below
# (temp 1.0, template-default xhigh effort), 600 tokens, mean of 3 seeds.
#
#   | config                    | greedy    | accepted | think     | accepted |
#   |---------------------------|-----------|----------|-----------|----------|
#   | SPEC=0                    | 28.9 t/s  | --       | 25.6 t/s  | --       |
#   | DRAFT_MAX=1               | 38.3      | 86%      | --        | --       |
#   | DRAFT_MAX=2 (default)     | 46.0 1.59x| 82%      | 32.8 1.28x| 59%      |
#   | DRAFT_MAX=3               | 46.9      | 69%      | 32.1      | 48%      |
#   | DRAFT_MAX=4               | 48.8      | 67%      | 31.0      | 41%      |
#   | DRAFT_MAX=5               | 47.6      | 60%      | 31.9      | 36%      |
#
# Unsloth's guide suggests 5 and their MTP README suggests 2. On sampled
# thinking output, which is what this server mostly produces, 2 is fastest,
# and past 2 the extra drafts are mostly rejected. On greedy text 3-5 run
# 2-6% faster, which is within run-to-run noise (the same config measured
# 42.7 and 46.0 across two boots). A sampled non-thinking run (temp 1.0, same
# prompt as greedy) put DRAFT_MAX=2 at 38.7 t/s vs 26.3 without the head.
#
# CONCURRENCY. Unsloth measured MTP as a net LOSS (0.81-0.87x) at concurrency 8
# on a B200. That did not reproduce on the GB10, where decode is bound by
# memory bandwidth and verifying 3 tokens per slot costs little more than
# verifying 1. Measured greedy, no thinking, 300 tokens per stream, aggregate
# t/s including prefill (1/2/4 streams at PARALLEL=4, 8 streams at PARALLEL=8):
#
#   | streams | SPEC=0   | DRAFT_MAX=2 | per-stream (MTP) |
#   |---------|----------|-------------|------------------|
#   | 1       | 25.4     | 43.6 (1.72x)| 44.7             |
#   | 2       | 37.5     | 52.1 (1.39x)| 27-31            |
#   | 4       | 52.4     | 60.2 (1.15x)| 16-18            |
#   | 8       | 62.9     | 78.1 (1.24x)| 10-12            |
#
# The gain shrinks as concurrency rises. At sampled-thinking acceptance (~59%
# rather than ~80%) it will shrink further, and it may turn into a loss at high
# concurrency (unmeasured). For batch-heavy serving, compare SPEC=0 at your own
# PARALLEL before choosing.
#
# Prefill (measured, 1 stream): 7.6k-token prompt at 407 t/s, 30.4k at 688 t/s.
# A long prompt in one slot throttles decode in the other while it runs.
# Model load is 22-36 s with the weights warm in page cache (right after
# download). A cold load after a reboot is unmeasured and will be longer.
#
# SAMPLING / REASONING. Server-side defaults follow Qwen's published settings,
# and clients can still override them per request:
#   thinking (default): temp 1.0, top_p 0.95, top_k 20, min_p 0, presence 0.0, repeat 1.0
#   REASONING=none:     temp 0.7, top_p 0.80, top_k 20, min_p 0, presence 1.5, repeat 1.0
# The embedded template only accepts reasoning_effort xhigh (its default),
# medium or low ("high" is mapped to xhigh). Effort "none" does NOT exist there
# -- `{"reasoning_effort":"none"}` is a 500 "Unexpected reasoning effort none".
# Non-thinking is `enable_thinking: false`, which is what REASONING=none sets
# server-wide via `--reasoning off`. Per request, pass
# `chat_template_kwargs: {"enable_thinking": false}` or `{"reasoning_effort": "low"}`.
# xhigh and low add a system line ("Reasoning effort is set to xhigh. Please
# think carefully..."), and medium adds nothing. REASONING=default injects
# nothing, so the model's own default (xhigh) stands. On one short factual
# prompt (single samples), xhigh used 214 completion tokens, medium 461 and low
# 247, so there is no sign here of runaway thinking at the default. Long
# prompts are unmeasured.
#
# Tool calling (verified): a get_weather call comes back as a standard OpenAI
# tool_calls entry with finish_reason tool_calls. The tool result round-trips
# into a normal answer. Reasoning arrives in reasoning_content and counts
# against max_tokens. Vision (verified): an image request is described
# correctly with the head active (draft acceptance logged).
server_args=(
  -ngl "${GPU_LAYERS}"
  --ctx-size "${CTX}"
  --flash-attn "${FLASH_ATTN}"
  --cache-type-k "${CACHE_TYPE}"
  --cache-type-v "${CACHE_TYPE}"
  --parallel "${PARALLEL}"
  --jinja
  --top-k 20
  --min-p 0.0
  --repeat-penalty 1.0
)

case "$REASONING" in
  default)          server_args+=(--temp 1.0 --top-p 0.95 --presence-penalty 0.0) ;;
  low|medium|xhigh) server_args+=(--reasoning-effort "$REASONING"
                                  --temp 1.0 --top-p 0.95 --presence-penalty 0.0) ;;
  none)             server_args+=(--reasoning off
                                  --temp 0.7 --top-p 0.80 --presence-penalty 1.5) ;;
  *) echo "REASONING must be default|xhigh|medium|low|none, got '$REASONING'" >&2; exit 2 ;;
esac

if [[ -n "$TARGET_HOST" ]]; then
  server_args=(-m "$(to_ctr "$TARGET_HOST")" "${server_args[@]}")
else
  server_args=(-hf "${REPO}:${QUANT}" "${server_args[@]}")
fi

# -m never auto-loads an mmproj and -hf does, so be explicit both ways.
if [[ -n "$MMPROJ_CTR_PATH" ]]; then
  server_args+=(--mmproj "$MMPROJ_CTR_PATH")
else
  server_args+=(--no-mmproj)
fi

if [[ "${SPEC:-1}" == 1 ]]; then
  server_args+=(
    -md "$DRAFT_CTR_PATH"
    --spec-type draft-mtp
    -ngld 999
    --spec-draft-n-max "${DRAFT_MAX}"
  )
fi

# --- docker run ------------------------------------------------------------
# One shared host model store (HF_HOME). With nothing cached, -hf pulls the
# target into llama.cpp's cache under it (that fallback path is untested here;
# the measured runs all served from the hub cache).
run_flags=(--gpus all --ipc=host -p "${PORT}:8080"
           -e "HF_TOKEN=${HF_TOKEN:-}"
           -v "${HF_HOME}:/root/.cache/huggingface"
           -e "HF_HOME=/root/.cache/huggingface"
           -e "LLAMA_CACHE=/root/.cache/huggingface/llama.cpp")

if [[ "${DETACH:-0}" == 1 ]]; then
  run_flags+=(-d --name qwen3.8-flash-next --restart unless-stopped)
else
  run_flags+=(--rm -it)
fi

echo ">> serving ${REPO}:${QUANT}${TARGET_HOST:+ (cached)}  (spec=${SPEC:-1} draft_max=${DRAFT_MAX} vision=${MMPROJ} reasoning=${REASONING} kv=${CACHE_TYPE} fa=${FLASH_ATTN})"
echo ">> ${PARALLEL} slot(s) sharing ${CTX} ctx"
echo ">> http://localhost:${PORT}  (OpenAI-compatible: /v1/chat/completions , Web UI at /)"
set -x
exec docker run "${run_flags[@]}" "$IMAGE" "${server_args[@]}" "$@"
