#!/usr/bin/env bash
# Serve ggml-org/Laya-GGUF (Laya -- a 421M ModernBERT-large *decision model*,
# by default the Q8_0 quant) with the from-source llama.cpp server image on the
# DGX Spark (GB10 / sm_121a).
#
# Laya does not generate text. It answers typed questions (choice / score /
# noul) about a `state` with calibrated probabilities, via the TypeSafe-style
# `POST /v1/systemone` endpoint (ggml-org/llama.cpp#29818). Chat completions
# are not served by this model.
#
#   curl localhost:8081/v1/systemone -H 'Content-Type: application/json' -d '{
#     "state": "I was charged twice for order #4471 and want my money back.",
#     "questions": {
#       "intent":  {"type": "choice", "instructions": "What does the customer want?",
#                   "criteria": {"refund": "wants money back", "track": null, "other": null}},
#       "urgent":  {"type": "noul",   "instructions": "Does this need a human within the hour?"},
#       "mood":    {"type": "score",  "instructions": "How frustrated is the customer?",
#                   "criteria": ["calm", "annoyed", "angry"]}
#     }}'
#
# Small enough to sit next to the big chat model on this box: it defaults to
# PORT 8081 so it can run beside run-qwen3.8-flash-next.sh (8080). See the
# MEASURED notes below for what that costs each side.
#
# REQUIRES llama.cpp WITH ggml-org/llama.cpp#29818 (first in release b11361).
# Measured on b11881 (image fa3645fe).
#
# Usage:
#   ./run-laya.sh                       # foreground (Ctrl-C to stop)
#   DETACH=1 ./run-laya.sh              # background server, restarts on boot
#   QUANT=BF16 ./run-laya.sh            # unquantized (0.84 GB instead of 0.45 GB)
#   PARALLEL=16 ./run-laya.sh           # more concurrent questions
#   ./run-laya.sh --threads 4           # append/override any llama-server flag
#
# Env: IMAGE, PORT (host), QUANT, PARALLEL, UBATCH, GPU_LAYERS, ALIAS, HF_TOKEN,
#      HF_HOME, DETACH.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-llama-spark:latest}"
REPO="ggml-org/Laya-GGUF"
ALIAS="${ALIAS:-Laya}"                 # API model id; without it, -m reports the file name
QUANT="${QUANT:-Q8_0}"                 # Q8_0 0.45 GB | BF16 0.84 GB (the only two published)
PORT="${PORT:-8081}"
PARALLEL="${PARALLEL:-8}"              # each question of a request is its own task, so slots
                                       # = questions decided concurrently. No KV cache (Laya is
                                       # an encoder), so slots are ~free: 4 -> 8 changed GPU use
                                       # by nothing measurable, and a 5-question request fell
                                       # from 29 to 21 ms because all 5 run in one pass
UBATCH="${UBATCH:-8192}"               # ModernBERT's trained context. Laya is non-causal: a
                                       # whole prompt (state + question + options) is read in
                                       # ONE ubatch, so this is the longest prompt it accepts
GPU_LAYERS="${GPU_LAYERS:-999}"        # 999 = offload every layer (whole model on GPU)
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

# --- resolve what is already in the shared hub cache ------------------------
# Same logic (and cache) as run.sh: if the GGUF is already in $HF_HOME/hub
# (e.g. `hf download ggml-org/Laya-GGUF Laya-Q8_0.gguf`), serve it in place
# with -m rather than re-downloading into llama.cpp's own cache.
resolve_cached_gguf() {
  local repo="$1" pat="$2" cache_repo snap
  cache_repo="$HF_HOME/hub/models--${repo//\//--}"
  [[ -d "$cache_repo/snapshots" ]] || return 0
  snap=""
  [[ -f "$cache_repo/refs/main" ]] && snap="$cache_repo/snapshots/$(<"$cache_repo/refs/main")"
  [[ -d "$snap" ]] || snap="$(find "$cache_repo/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
  [[ -d "$snap" ]] || return 0
  find -L "$snap" -type f -iname "$pat" 2>/dev/null | sort | head -n1
}

TARGET_HOST="$(resolve_cached_gguf "$REPO" "*${QUANT}*.gguf")"

# --- llama-server options --------------------------------------------------
# The server spots `{arch}.decision.type = laya` in the GGUF and switches to
# embedding mode itself (no pooling, n_batch clamped to n_ubatch), and turns
# the host-RAM prompt cache off (decision states rarely repeat). So the only
# load-bearing choices are the batch size and the per-slot context.
#
# --ctx-size is TOTAL across slots, so it scales with PARALLEL to give every
# slot the full UBATCH window. A prompt longer than UBATCH is rejected, not
# truncated -- Laya only trims long questions/options, never the state. The
# limit is PER QUESTION (state + that question + its options): a state of
# ~7.9k tokens still works, 8671 returns HTTP 500 "input (8671 tokens) is too
# large to process. increase the physical batch size".
#
# MEASURED on the Spark (Q8_0, PARALLEL=8, beside Qwen3.8-Flash-Next):
#   - memory: 1.2 GiB GPU (nvidia-smi), container RSS ~0.9 GiB; load < 1 s
#   - accuracy: the PR #29818 reference request (5 questions, 414 tokens)
#     matches the reference to within 0.003 on every probability
#   - latency: 5-question / 414-token request 21 ms (Qwen idle), 50-80 ms
#     while Qwen is decoding; a ~3.6k-token question ~0.4 s, ~7.9k ~0.9 s
#   - throughput: 8 clients x 5 questions: ~32 req/s (~160 decisions/s),
#     every client served evenly, worst request 0.27 s, Qwen decoding meanwhile
#
# CAVEAT -- long states under sustained concurrency. When every prompt nearly
# fills UBATCH, only one fits per batch, and llama-server keeps refilling the
# batch from the same slots: with 4 clients looping on ~7k-token states, two
# ran at 2.1 s/request while the other two waited ~32 s. More slots do not
# help (same at PARALLEL=8). Short states (the normal case) are unaffected.
server_args=(
  --alias "${ALIAS}"
  -ngl "${GPU_LAYERS}"
  --ctx-size "$((UBATCH * PARALLEL))"
  --batch-size "${UBATCH}"
  --ubatch-size "${UBATCH}"
  --parallel "${PARALLEL}"
  --flash-attn on
)

if [[ -n "$TARGET_HOST" ]]; then
  server_args=(-m "/root/.cache/huggingface/${TARGET_HOST#"$HF_HOME"/}" "${server_args[@]}")
else
  server_args=(-hf "${REPO}:${QUANT}" "${server_args[@]}")
fi

# --- docker run ------------------------------------------------------------
run_flags=(--gpus all --ipc=host -p "${PORT}:8080"
           -e "HF_TOKEN=${HF_TOKEN:-}"
           -v "${HF_HOME}:/root/.cache/huggingface"
           -e "HF_HOME=/root/.cache/huggingface"
           -e "LLAMA_CACHE=/root/.cache/huggingface/llama.cpp")

if [[ "${DETACH:-0}" == 1 ]]; then
  run_flags+=(-d --name laya --restart unless-stopped)
else
  run_flags+=(--rm -it)
fi

echo ">> serving ${REPO}:${QUANT}${TARGET_HOST:+ (cached)}  (${PARALLEL} slot(s), ${UBATCH}-token prompts)"
echo ">> http://localhost:${PORT}/v1/systemone  (decision API; no chat completions)"
set -x
exec docker run "${run_flags[@]}" "$IMAGE" "${server_args[@]}" "$@"
