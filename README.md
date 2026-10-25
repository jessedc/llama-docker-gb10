# llama.cpp on DGX Spark (GB10) — Dockerized server

From-source, reproducible Docker build of the llama.cpp **server** for the
NVIDIA DGX Spark (GB10 Grace Blackwell, `sm_121a`), built with CUDA 13.

The approach: pin a known-good CUDA base image, recompile the GPU kernels for
this exact chip, and record the pins in `build.lock` for reproducible rebuilds.

## Prerequisites

- An NVIDIA DGX Spark (GB10) with the GPU driver installed (`nvidia-smi` works).
- Docker, plus the **NVIDIA Container Toolkit** — `--gpus all` won't work without
  it. Quick check: `docker run --rm --gpus all nvidia/cuda:13.0.3-runtime-ubuntu24.04 nvidia-smi`
  should print the GPU. If it errors, install the toolkit and restart Docker.
- Network access at build time (clones llama.cpp; fetches the Web UI unless `--no-ui`).

## Why build instead of pulling a prebuilt image

There is no reliable prebuilt GB10 image to `docker pull`:

- The official `ghcr.io/ggml-org/llama.cpp` CUDA images default to **CUDA 12**,
  which has **no `sm_121` support at all** — they won't run on GB10.
- The `-cuda13` variants are built with a generic arch list, are *not* GPU-CI
  tested, and are not tuned for `sm_121a`.

So we compile upstream llama.cpp from a pinned commit `FROM nvidia/cuda:13.0.x-devel`
with `-DCMAKE_CUDA_ARCHITECTURES=121a`, then ship the binary on a slim
`-runtime` image. The GPU driver is injected at run time via `--gpus all`.

## Layout

| File          | Purpose |
|---------------|---------|
| `Dockerfile`  | Two-stage build (devel → runtime), server target only. |
| `build.sh`    | Resolve + pin base digest & llama.cpp commit, build, write `build.lock`. |
| `run.sh`      | Serve any HF repo or a local GGUF with `--gpus all` (the generic runner). |
| `run-gemma4-12b.sh` | Pinned runner for `unsloth/gemma-4-12b-it-GGUF` (default `UD-Q4_K_XL`). |
| `run-zeta-2.sh` | Pinned runner for `bartowski/zed-industries_zeta-2-GGUF` (default `Q8_0`). |
| `run-deepseek-v4-flash.sh` | Pinned runner for `unsloth/DeepSeek-V4-Flash-0731-GGUF` (default `UD-IQ2_M`), with DSpark speculative decoding. |
| `run-muse-glimmer.sh` | Pinned runner for `unsloth/Muse-Glimmer-30B-GGUF` (default `UD-Q6_K_XL`), with vision + DFlash speculative decoding. |
| `run-qwen3.8-flash-next.sh` | Pinned runner for `unsloth/Qwen3.8-Flash-Next-GGUF` (default `UD-IQ4_XS`), with vision + MTP speculative decoding. |
| `run-laya.sh` | Pinned runner for the `ggml-org/Laya-GGUF` decision model (default `Q8_0`) on port 8081, serving `/v1/systemone`. Fits beside Qwen. |
| `build.lock`  | Generated pins for `./build.sh --reproduce`. |

## Build

```bash
./build.sh                 # latest llama.cpp master HEAD
./build.sh --ref b11881    # a specific tag/branch/commit (current pin; or a 40-char SHA)
./build.sh --reproduce     # rebuild exactly what build.lock records
./build.sh --no-ui         # skip the embedded Web UI (no build-time HF fetch)
```

First build compiles the CUDA kernels (~several minutes on the Spark); `ccache`
is mounted as a BuildKit cache so rebuilds are fast.

## Serve

```bash
# Pull + serve a GGUF straight from Hugging Face (cached under ~/.cache/huggingface)
./run.sh ggml-org/gemma-3-4b-it-GGUF

# Serve a local GGUF (its directory is mounted read-only)
./run.sh /path/to/models/<model>.gguf

# Extra llama-server flags pass straight through
./run.sh ggml-org/gemma-3-4b-it-GGUF --ctx-size 32768 --parallel 4

# Background server mode (auto-restart)
DETACH=1 ./run.sh ggml-org/gemma-3-4b-it-GGUF
```

Then: OpenAI-compatible API at `http://localhost:8080/v1/chat/completions`, and
the Web UI at `http://localhost:8080/`. Quick check it's up:

```bash
curl localhost:8080/health                       # -> {"status":"ok"}
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}]}'
```

### Reusing GGUFs already in the shared cache

If a GGUF for a repo is already in the shared HF cache (e.g. you ran
`hf download ggml-org/gemma-3-4b-it-GGUF gemma-3-4b-it-Q8_0.gguf`), `run.sh`
detects it and serves it **in place** (`-m`) instead of re-downloading — so the
same file stays usable by any other tool reading the HF cache. Pin a quant with
`repo:QUANT` (e.g. `./run.sh ggml-org/gemma-3-4b-it-GGUF:Q8_0`); sharded models
resolve to the first shard automatically.

### Pinned per-model runners

For models we serve regularly there are dedicated scripts that pin the repo,
quant, and known-good llama-server flags so you don't have to remember them.
They share the same env vars (`PORT`, `QUANT`, `GPU_LAYERS`, `DETACH`, …) and
pass any extra flags straight through to `llama-server`:

```bash
./run-gemma4-12b.sh                    # unsloth/gemma-4-12b-it-GGUF  (UD-Q4_K_XL)
QUANT=UD-Q5_K_XL ./run-gemma4-12b.sh   # override the quant
./run-zeta-2.sh                        # bartowski/zed-industries_zeta-2-GGUF (Q8_0)
DETACH=1 ./run-zeta-2.sh               # background server, restarts on boot
./run-deepseek-v4-flash.sh             # unsloth/DeepSeek-V4-Flash-0731-GGUF (UD-IQ2_M)
./run-qwen3.8-flash-next.sh            # unsloth/Qwen3.8-Flash-Next-GGUF (UD-IQ4_XS)
./run-muse-glimmer.sh                  # unsloth/Muse-Glimmer-30B-GGUF (UD-Q6_K_XL)
./run-laya.sh                          # ggml-org/Laya-GGUF (Q8_0) decision model, port 8081
```

#### DeepSeek-V4-Flash-0731 — 284B MoE + DSpark speculative decoding

`run-deepseek-v4-flash.sh` serves DeepSeek-V4-Flash-0731 (284B total / 38B
active — 43 layers, 256 routed + 1 shared expert, 6 experts per token, 1M
native context) on this single box, and pulls two files from the one repo:

| Part | File | Size | Notes |
|------|------|------|-------|
| Target model | `UD-IQ2_M/…-UD-IQ2_M-*.gguf` (3 shards) | 84.68 GiB | selected by `-hf repo:QUANT` |
| DSpark drafter | `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` | 10.15 GiB | repo root; named explicitly with `-md` |

Unsloth puts the drafter at the repo **root** so one copy serves every quant
folder and llama.cpp auto-discovers it. The script names it with `-md` anyway:
an explicit path never depends on `-hf` having pulled the repo's auxiliary
files, and costs nothing when it has.

Both files are resolved out of the shared hub cache first, using the same
`resolve_cached_gguf` logic as `run.sh` — so

```bash
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \
  --include "UD-IQ2_M/*" --include "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
```

pre-stages the ~95 GiB into `$HF_HOME/hub`, and the script then serves it in
place with `-m` (first shard; llama-server finds the other two) instead of
re-downloading into llama.cpp's separate cache under `$HF_HOME/llama.cpp`. With
nothing cached it falls back to `-hf` and pulls it itself.

**DSpark** is DeepSeek's own speculative module, extracted from the 0731
checkpoint (25 FP8 projections, BF16 Markov/confidence heads, MXFP4 routed
experts; `general.architecture = dflash`). It requires `--spec-type draft-dspark`
— not the `draft-simple` path used by ordinary same-family draft models.
`--spec-draft-n-max 3` is Unsloth's measured default and this script's; at load
it reports `block_size=5, mask_token_id=128799, n_extract=3`, matching the
`dspark_*` fields in the source `config.json`.

**Measured on the Spark** (image `ba360efe` / b10375, `UD-IQ2_M`, 300-token
single stream, `temperature 0`, warm, 21-token prompt):

| Config | Decode | Draft accepted |
|--------|--------|----------------|
| `SPEC=0` (no drafter) | 20.57 t/s | — |
| `SPEC=1`, f16 KV, `-fa auto` | 30.01 t/s (**1.46×**) | 180/354 (51%) |
| `SPEC=1`, q8_0 KV, `-fa on` (default) | **31.58 t/s** (**1.54×**) | 184/343 (54%) |

Model load is ~80–100 s; resident is ~102 GiB of the 121 GiB in every config —
weights dominate, so the q8_0 KV saving buys context headroom rather than a
smaller footprint. For reference, `tekosML`'s GB10-tuned IQ2XXS build measures
16.9 t/s target-only, and Unsloth quotes ~2× for DSpark on a B200.

Tool calling works: DeepSeek's DSML blocks parse into standard OpenAI
`tool_calls` with `finish_reason: tool_calls`. Reasoning arrives in a separate
`reasoning_content` field and **counts against `max_tokens`** — a 300-token cap
on a hard prompt is consumed entirely by thinking and leaves `content` empty.

##### Why llama.cpp and not vLLM for this model

The FP8 source checkpoint is **155 GiB** and does not fit this machine. Of every
published re-quantization, only three artifacts fit a 128 GB single-Spark:

| Artifact | Size | Runtime | Verdict |
|---|---|---|---|
| `unsloth/…-GGUF:UD-IQ2_M` (+ drafter) | 84.7 + 10.2 GiB | llama.cpp | **chosen** — native `deepseek4` + DSpark, no plugin |
| `tekosML/…-GGUF-GX10` | 80.8 GiB | llama.cpp | GB10-tuned imatrix; measured 16.9 t/s target-only, no drafter |
| `rdtand/…-gridbook-87GB-spark-vllm` | 81.2 GiB | vLLM | needs out-of-tree `gridbook` plugin; CUDA-graph bug on vLLM 0.27+; **DSpark sidecar excluded** |
| NVFP4 / W4A16 / GPTQ-Int4 / AutoRound | 142–170 GiB | vLLM | do not fit |

vLLM v0.27.1 *does* register `DeepseekV4ForCausalLM` and `DSparkDraftModel`, so
the engine is not the blocker — the weights are. Since DSpark is the biggest
decode lever available and only the GGUF path can use it here, this model is
served with llama.cpp.

##### Load-bearing flags

- **`--ctx-size` must be explicit.** `deepseek4.context_length` in the GGUF is
  `1048576`, and llama.cpp's default (`--ctx-size 0`) means *take it from the
  model* — which tries to size a million-token KV pool on top of 95 GiB of
  weights and dies. `CTX` defaults to `524288`, and it is the **total** budget
  llama.cpp splits across slots: the default `PARALLEL=2` gives each slot
  `262144`. Raise `CTX` in step with `PARALLEL`, or each slot's budget shrinks.
  Context is cheap here (32768 → 262144 measured at only +977 MiB, since most
  blocks use a 128-token sliding window), so the limit is prefill time, not memory.
- **`--jinja` is not optional.** The base checkpoint ships **no chat template at
  all** — only a programmatic encoder (`encoding/encoding_dsv4.py`). The Jinja
  port Unsloth embedded in the GGUF is the only thing emitting the right control
  tokens, and tool calling depends on it. Tool calls come back as DeepSeek DSML
  blocks, handled by the build's DeepSeek V3.2 parser.
- **`REASONING` defaults to `none`,** which injects nothing — the template's own
  default. `high` and `max` prepend maximum-deliberation system prompts
  ("leave absolutely nothing to chance…") that can run a trivial prompt for many
  minutes. vLLM's `deepseek_v4` path falls into exactly this trap by defaulting
  to `high`; here we simply don't.
- **`FLASH_ATTN=on` + `q8_0` KV is measured, not assumed.** The concern was that
  DeepSeek-V4 attention carries a sparse indexer (`index_topk 512` over 64 index
  heads) alongside the MLA-style compressed latent path, and might reject flash
  attention on an arch this new. It doesn't — b10375 accepts `-fa on` with a
  `q8_0` K/V cache, and that's both the fastest configuration measured and half
  the KV bytes, so it's the default. `FLASH_ATTN=auto CACHE_TYPE=f16` is the
  fallback if a future build regresses.

```bash
./run-deepseek-v4-flash.sh                  # IQ2_M + DSpark, 2 slots x 256K ctx
PARALLEL=1 CTX=262144 ./run-deepseek-v4-flash.sh   # single slot, fastest single stream
QUANT=UD-IQ1_M ./run-deepseek-v4-flash.sh   # 80.9 GiB instead of 84.7
SPEC=0 ./run-deepseek-v4-flash.sh           # no drafter, frees ~10 GiB
REASONING=high ./run-deepseek-v4-flash.sh   # none | high | max
CACHE_TYPE=f16 FLASH_ATTN=auto ./run-deepseek-v4-flash.sh   # pre-measurement fallback
```

Quant sizes in this repo, for fitting against the 121 GiB the Spark actually
has (add 10.15 GiB whenever `SPEC=1`): `UD-IQ1_S` 76.9 · `UD-IQ1_M` 80.9 ·
`UD-IQ2_XXS` 84.6 · **`UD-IQ2_M` 84.7** · `UD-Q2_K_XL` 90.2 · `UD-IQ3_XXS`
97.1 GiB. Unsloth suggests `UD-IQ3_XXS` for 128 GB machines, but 97.1 + 10.2 =
107 GiB leaves no room for the KV pool once the desktop and page cache are
accounted for — `UD-IQ2_M` is the honest ceiling with DSpark on.

Requires a llama.cpp build **≥ b10269**: b10228 added DeepSeek-V4 DSpark, but
b10259–b10268 advertise `draft-dspark` and then abort while loading the drafter.
Measured on b10375; the current `build.lock` pin (b11881) is clear of that window too.

> **Memory**: weights + drafter are ~95 GiB of the shared 121 GiB. Nothing else
> substantial can be resident — stop any other GPU-heavy containers first
> (`docker ps`, then `docker stop <name>`).

#### Muse Glimmer 30B — vision + speculative decoding

`run-muse-glimmer.sh` serves Meta's Muse Glimmer 30B (dense 30B + 1.8B
perception encoder, 131k native context) and pulls in three files from the one
repo:

| Part | File | Size | Notes |
|------|------|------|-------|
| Text model | `Muse-Glimmer-30B-UD-Q6_K_XL.gguf` | ~25 GB | selected by `-hf repo:QUANT` |
| Perception encoder | `mmproj-...-Q8_0.gguf` | ~2.0 GB | loaded automatically; `--no-mmproj` opts out |
| DFlash drafter | `dflash-kquant.gguf` | ~1.6 GB | must be named explicitly with `-md` |

`-hf` downloads all three — it pulls the repo's auxiliary files alongside the
model. The drafter still has to be *named* on the command line, and it can't be
named with `-hfd/--hf-repo-draft`, because `dflash-kquant` is **not a valid
Hugging Face quantization tag** (that endpoint 400s), so the lookup would fall
back to the wrong file. The script therefore points `-md` at the copy in
llama.cpp's cache, fetching it by URL itself only if it isn't there yet.

DFlash is a *block-diffusion* drafter: it proposes a whole block of tokens per
forward pass and the 30B verifies them in parallel, so decode speeds up with
identical output. It requires `--spec-type draft-dflash` — not the `draft-simple`
path used for ordinary same-family draft models.

**Measured on the Spark** (Q6, 400-token generation, `temperature 0`):

| Config | Decode | Draft acceptance |
|--------|--------|------------------|
| `SPEC=0` (off) | 8.24 t/s | — |
| `SPEC=1`, `DRAFT_MAX=4` (default) | **17.68 t/s** (**2.1×**) | 0.40, mean len 2.58 |
| `SPEC=1`, `DRAFT_MAX=8` | 16.67 t/s | 0.22, mean len 2.71 |
| `SPEC=1`, `DRAFT_MAX=16` | 16.84 t/s | 0.13, mean len 2.81 |

`DRAFT_MAX=4` measured fastest, so that's the default — a longer draft window
raises the mean accepted length slightly but wastes more verification work.

Reasoning text comes back in a separate `reasoning_content` field, not in
`content`. It counts against `max_tokens`, so a small `max_tokens` can be fully
consumed by reasoning and leave `content` empty.

```bash
./run-muse-glimmer.sh                    # Q6 + vision + speculative decoding, 4 slots
QUANT=UD-Q4_K_XL ./run-muse-glimmer.sh   # 16 GB instead of 25 GB
SPEC=0 ./run-muse-glimmer.sh             # disable speculative decoding
MMPROJ=0 ./run-muse-glimmer.sh           # text-only (skips the 2.0 GB encoder)
REASONING=high ./run-muse-glimmer.sh     # low | medium | high | xhigh
PARALLEL=1 ./run-muse-glimmer.sh         # single slot
```

**Concurrency.** `--ctx-size` in llama.cpp is the *total* budget split across
slots, so raising `--parallel` against a fixed total silently shrinks every
slot — 4 slots against `-c 131072` leaves each request only 32768. `CTX` in
this script is therefore **per slot** and the total is multiplied out, so
`PARALLEL=4` really does mean 4 × 131072.

Four slots is the default because the extra ones are free until used:

| Load | Per request | Aggregate |
|------|-------------|-----------|
| 1 request | 15.7 t/s | 15.7 t/s |
| 4 concurrent | 11.7–13.1 t/s | **~49 t/s (3.1×)** |

A lone request measured the same at `PARALLEL=1`, `2`, and `4`, and the extra
slots added no measurable memory — the model's alternating sliding-window
layers (window 2048) keep the KV cache small, so only about half the layers
hold the full context.

Reasoning strength is set via `--chat-template-kwargs '{"reasoning_strength":...}'`
and defaults to `low`. Requires a llama.cpp build **≥ b10353** (upstream #26841);
measured on b10375, and the current `build.lock` pin is newer.

> **Memory**: the Spark's 121 GB of unified memory is shared with anything else
> running on the box. Another large model already resident will not leave room
> for this one — stop other GPU-heavy containers first (`docker ps`, then
> `docker stop <name>`).

#### Qwen3.8-Flash-Next — 125B MoE + vision + MTP speculative decoding

`run-qwen3.8-flash-next.sh` serves Qwen3.8-Flash-Next (125B MoE, arch
`qwen4exp`: 48 layers, 512 routed experts with 10 per token, 262,144 native
context, hybrid thinking, multimodal) and pulls three files from the one repo:

| Part | File | Size | Notes |
|------|------|------|-------|
| Target model | `UD-IQ4_XS/…-UD-IQ4_XS-*.gguf` (3 shards) | 87.24 GiB | 61.2 GiB goes to the GPU; the 26.8 GiB per-layer-embedding table stays mmapped |
| MTP head | `MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` | 2.59 GiB | must be named explicitly with `-md` |
| Vision encoder | `mmproj-F16.gguf` | 0.84 GiB | `MMPROJ=0` opts out |

```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-IQ4_XS/*" \
  --include "MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf" --include "mmproj-F16.gguf"
```

This pre-stages all ~90 GiB into `$HF_HOME/hub`, and the script serves it in
place. Without it, the script falls back to `-hf` for the target and fetches the
head and encoder by URL (that path is untested).

**Needs llama.cpp with
[ggml-org/llama.cpp#29761](https://github.com/ggml-org/llama.cpp/pull/29761)**
("Qwen4Exp: add MTP", merged 2026-10-01). `build.lock` pins release `b11881`.
The earlier PR #28243 build this runner started on was closed unmerged.

**The MTP head** is the model's own multi-token-prediction layer. It must be a
**self-contained** head: mainline requires the head to carry its own token
embedding, so the repo's `shared-` heads (which borrowed it from the target on
the #28243 build) don't load. Unsloth re-exported the self-contained heads for
mainline on 2026-10-05. `Q4_K_M` is used rather than `Q8_0` (3.85 GiB; also at
the repo root for `-hf` auto-discovery), trading ~2 points of acceptance for
1.26 GiB of headroom. Two things to know:

- **`-md` is always passed explicitly.** Serving a pre-staged file with `-m`
  skips sidecar auto-discovery, so `--spec-type draft-mtp` alone silently runs
  without speculation. Check for `draft acceptance = …` in
  `docker logs qwen3.8-flash-next`.
- **The model id is `Qwen3.8-Flash-Next`** (`--alias`; override with `ALIAS=`).
  Without it, serving with `-m` reports the first shard's filename.

**Re-measured on b11696** (image `ceefd16d`, self-contained `Q4_K_M` head, same
method as below; 77,770 MiB GPU at boot):

| Config | Greedy | Accepted | Think | Accepted |
|--------|--------|----------|-------|----------|
| `SPEC=0` | 29.4 t/s | — | 29.2 t/s | — |
| `DRAFT_MAX=2` (default) | 43.5 t/s (**1.48×**) | 73% | **37.9 t/s (1.30×)** | 58% |

On **b11881** (image `fa3645fe`) the default config measured 44–46 t/s greedy
(300 tokens) with 77% of drafts accepted, at 77,704 MiB GPU. Tool calling,
MTP and vision all load and work unchanged.

**Measured on the Spark** with the earlier PR #28243 build (image `d1a92352`,
`shared-Q4_K_M` head, `UD-IQ4_XS`, q8_0 KV, `-fa on`, warm, single stream).
*Greedy* is temperature 0 with thinking off, 400 tokens.
*Think* is the server defaults (temperature 1.0, template-default `xhigh`
effort), 600 tokens, mean of 3 seeds.

| Config | Greedy | Accepted | Think | Accepted |
|--------|--------|----------|-------|----------|
| `SPEC=0` | 28.9 t/s | — | 25.6 t/s | — |
| `DRAFT_MAX=1` | 38.3 t/s | 86% | — | — |
| `DRAFT_MAX=2` (default) | 46.0 t/s (**1.59×**) | 82% | **32.8 t/s (1.28×)** | 59% |
| `DRAFT_MAX=3` | 46.9 t/s | 69% | 32.1 t/s | 48% |
| `DRAFT_MAX=4` | 48.8 t/s | 67% | 31.0 t/s | 41% |
| `DRAFT_MAX=5` | 47.6 t/s | 60% | 31.9 t/s | 36% |

`DRAFT_MAX=2` (the MTP README's default, not the guide's 5) is fastest on
sampled thinking output, which is what this server mostly produces. Drafting
past 2 is mostly rejected there. On greedy text, 3–5 are 2–6% faster, which is
within boot-to-boot noise (the same config measured 42.7 and 46.0 on two
boots). A sampled non-thinking run (temperature 1.0) measured 38.7 t/s with the
head vs 26.3 t/s without. Prefill: 407 t/s on a 7.6k-token prompt and 688 t/s
on 30.4k. Load: 22–36 s with the weights warm in page cache; a cold load is
unmeasured.

**Concurrency.** Unsloth measured MTP as a net loss (0.81–0.87×) at concurrency 8
on a B200. That did **not** reproduce here: the GB10 is bandwidth-bound, so
verifying 3 tokens per slot is cheap. The following is greedy, no thinking,
aggregate t/s including prefill:

| Streams | `SPEC=0` | `DRAFT_MAX=2` | Per stream (MTP) |
|---------|----------|---------------|------------------|
| 1 | 25.4 | 43.6 (1.72×) | 44.7 |
| 2 | 37.5 | 52.1 (1.39×) | 27–31 |
| 4 | 52.4 | 60.2 (1.15×) | 16–18 |
| 8 | 62.9 | 78.1 (1.24×) | 10–12 |

The gain shrinks with load, and sampled thinking accepts fewer drafts (59% vs
~80%), so heavy concurrent thinking workloads may still lose (unmeasured).
Defaults are therefore `PARALLEL=2`, `CTX=524288` (262,144 per slot), with MTP on.

**Memory.** Only part of the model keeps context-sized KV: two 12-layer caches,
~17 KiB/token at q8_0. Its recurrent layers hold a fixed 450 MiB of state per
slot. 8× the context (32k →
262k) costs ~5.9 GiB on the GPU (4.4 GiB KV plus a larger compute buffer). At
the defaults, nvidia-smi shows **77.5 GiB**. On top of that sits the 26.8 GiB
PLE table in page cache (visible in neither nvidia-smi nor free's `used`) and the
host-RAM prompt cache. That is ~104–106 GiB of 121 GiB.

The prompt cache is the one part that grows over time. At llama.cpp's
8,192 MiB default, the box had 8 GiB in swap after 9 days up. That's probably
the cache, though not proven. `CACHE_RAM` therefore defaults to **2048** MiB
(`--cache-ram`), which leaves ~6 GiB more for the PLE pages and for Laya beside
it (below). The cost is that a long prompt evicted from its slot gets
re-prefilled rather than restored from host RAM. `CACHE_RAM=8192` restores
llama.cpp's default, and `0` turns the cache off.
`PARALLEL=1 CTX=262144` saves 5.5 GiB, `MMPROJ=0` saves 1.1 GiB and `SPEC=0`
saves ~3 GiB.

**Reasoning.** The embedded template accepts `reasoning_effort` `xhigh` (the
default), `medium` or `low`. There is **no `"none"` effort**: sending it returns
HTTP 500 `Unexpected reasoning effort none`. Turn thinking off with
`enable_thinking: false` instead. `REASONING=none` does that server-wide
(`--reasoning off`) and switches the server sampler to Qwen's instruct settings
(temperature 0.7, top_p 0.80, presence_penalty 1.5). Per request, pass
`"chat_template_kwargs": {"reasoning_effort": "low"}` or
`{"enable_thinking": false}`. `REASONING=default` leaves the template's `xhigh`
in place. On one short prompt it didn't run long (214 completion tokens vs
medium's 461); long prompts are unmeasured.

Tool calling works: a `get_weather` call comes back as a standard OpenAI
`tool_calls` entry with `finish_reason: tool_calls`, and the tool result
round-trips into a normal answer. Vision works with the head active. Reasoning
arrives in `reasoning_content` and counts against `max_tokens`.

```bash
./run-qwen3.8-flash-next.sh                    # IQ4_XS + vision + MTP, 2 x 262k slots
REASONING=none ./run-qwen3.8-flash-next.sh     # instruct mode (no thinking)
REASONING=medium ./run-qwen3.8-flash-next.sh   # xhigh (template default) | medium | low
SPEC=0 ./run-qwen3.8-flash-next.sh             # no MTP head
MMPROJ=0 ./run-qwen3.8-flash-next.sh           # text-only
PARALLEL=1 CTX=262144 ./run-qwen3.8-flash-next.sh   # single slot, 5.5 GiB less
CACHE_RAM=0 ./run-qwen3.8-flash-next.sh        # no host-RAM prompt cache (default cap 2048 MiB)
DETACH=1 ./run-qwen3.8-flash-next.sh           # background server, restarts on boot
```

> **Memory**: ~104 GiB of the shared 121 GiB once the PLE table is paged in.
> Nothing else substantial can be resident: stop other GPU-heavy containers
> first (`docker ps`, then `docker stop <name>`). It also binds port 8080, like
> the other runners. The one measured exception is Laya (below), which is small
> enough to run beside it.

#### Laya — decision model (`/v1/systemone`), runs beside Qwen

[Laya](https://huggingface.co/blog/ggml-org/decision-models-in-llamacpp) is a
421M ModernBERT-large **decision model**. It doesn't generate text. It answers
typed questions about a `state` with probabilities, via the TypeSafe-style
`POST /v1/systemone` endpoint
([ggml-org/llama.cpp#29818](https://github.com/ggml-org/llama.cpp/pull/29818),
first released in `b11361`). `run-laya.sh` serves `ggml-org/Laya-GGUF:Q8_0`
(0.45 GB; `QUANT=BF16` is 0.84 GB) on **port 8081**, so it can sit next to a
chat model on 8080.

```bash
hf download ggml-org/Laya-GGUF Laya-Q8_0.gguf   # optional pre-stage; served in place
DETACH=1 ./run-laya.sh                           # container "laya", restarts on boot

curl localhost:8081/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "I was charged twice for order #4471 and want my money back.",
  "questions": {
    "intent": {"type": "choice", "instructions": "What does the customer want?",
               "criteria": {"refund": "wants money back", "track": null, "other": null}},
    "urgent": {"type": "noul",   "instructions": "Does this need a human within the hour?"},
    "mood":   {"type": "score",  "instructions": "How frustrated is the customer?",
               "criteria": ["calm", "annoyed", "angry"]}
  }}'
```

Question types: `choice` (pick one of up to 255 options), `score` (2–10 ordered
levels, returns the expected level), `noul` (probability of true). Each question
in a request is decided independently as its own task.

**Load-bearing flags.** The server reads `laya` from the GGUF's
`decision.type` and switches to embedding mode itself (no pooling, `n_batch`
clamped to `n_ubatch`, prompt cache off). Laya is non-causal, so **each
question's whole prompt (state + question + options) must fit in one ubatch**.
`UBATCH` defaults to 8192, ModernBERT's trained context. A ~7.9k-token state
works. At 8,671 tokens the server returns HTTP 500 `input … is too large to
process`: the state is rejected, not truncated (Laya only trims long questions
and options). `PARALLEL` defaults to 8 slots. Laya has no KV cache, so going
from 4 to 8 slots cost nothing measurable and cut a 5-question request from
29 to 21 ms.

**Measured** on b11881, Q8_0, `PARALLEL=8`, with Qwen3.8-Flash-Next resident:

| | Result |
|---|---|
| Memory | **1.2 GiB GPU** (nvidia-smi), ~0.9 GiB container RSS; loads in < 1 s |
| Accuracy | PR #29818's 5-question reference request (414 tokens): every probability within 0.003 of the reference |
| Latency, 5 questions / 414 tokens | 21 ms (Qwen idle), 50–80 ms while Qwen decodes |
| Latency, one long question | ~0.4 s at ~3.6k tokens, ~0.9 s at ~7.9k |
| Throughput | 8 clients: ~32 req/s (~160 decisions/s), even across clients, worst 0.27 s |

**Running it beside Qwen3.8-Flash-Next: yes.** Memory is not the problem.
Laya's 1.2 GiB takes system `MemAvailable` from ~11 to ~10 GiB with Qwen at its
defaults (77.7 GiB GPU plus the paged PLE table). Qwen's `CACHE_RAM=2048` cap
keeps its prompt cache from eating that margin over time. To free more, see
Qwen's memory options above: `PARALLEL=1 CTX=262144` (−5.5 GiB) and
`MMPROJ=0` (−1.1 GiB). Compute *is* shared: both
run on the one GPU, so Laya load slows Qwen decode (greedy, 300 tokens):

| Laya load | Qwen decode |
|---|---|
| idle | 44.1 t/s |
| 5 req/s × 5 questions (25 decisions/s) | 39.5 t/s (−10%) |
| saturated, short requests (~70 decisions/s) | 28.3 t/s (−36%) |
| saturated, 4 clients with ~7k-token states | 12–27 t/s |

A classifier sitting in front of a chat model at typical rates costs Qwen ~10%.
Bulk scoring of long documents is better run while Qwen is quiet.

**Caveat: long states under sustained concurrency.** When every prompt nearly
fills `UBATCH`, only one fits per batch, and llama-server keeps refilling it
from the same slots. In testing, 4 clients looped on ~7k-token states: two ran
at 2.1 s per request while the other two waited ~32 s. `PARALLEL=8` didn't help.
Short states are unaffected.

### Useful env vars (see `run.sh` header)

`IMAGE`, `PORT` (8080), `GPU_LAYERS` (999=all), `HF_TOKEN`, `HF_HOME`, `DETACH`.

`-hf` downloads go into the standard Hugging Face cache: `HF_HOME` defaults to
`~/.cache/huggingface`, shared with any other HF tooling on the host. llama.cpp's
flat `-hf` cache lands in a `llama.cpp/` subdir of it — its layout differs from
the HF hub `models--org--repo` layout, so files aren't deduped across the two,
but all models stay under one directory.

## Notes

- **Current pin**: release tag **`b11881`** (`fa3645feb`, 2026-10-25), which
  includes mainline Qwen3.8-Flash-Next MTP
  ([ggml-org/llama.cpp#29761](https://github.com/ggml-org/llama.cpp/pull/29761))
  and the `/v1/systemone` decision-model API
  ([#29818](https://github.com/ggml-org/llama.cpp/pull/29818)). Qwen3.8-Flash-Next
  and Laya are verified on it.
- **Previous pins**, still tagged locally for rollback
  (`IMAGE=llama-spark:<tag> ./run-….sh`): `ceefd16d` (release `b11696`),
  `d1a92352` (head of the closed,
  unmerged PR #28243; the only build that loads the `shared-` MTP heads) and
  `ba360efe` (release `b10375`). The DeepSeek-V4-Flash, Muse Glimmer, Gemma and
  Zeta runners were measured on `b10375` and have not been re-verified since. DeepSeek-V4-Flash DSpark
  needs ≥ `b10269`.
- **Verified baseline**: bare-metal build `c1304d7b2 (9671)` ran Qwen3.6-35B-A3B
  Q8 on the GB10 at ~697 t/s prefill / ~48 t/s decode, all layers on CUDA — this
  image reproduces that build inside a container.
- The image build needs network for the embedded Web UI assets (Hugging Face
  bucket `ggml-org/llama-ui`). Use `--no-ui` for a hermetic, API-only image.
- Unified memory: GB10 shares 128 GB between CPU and GPU, so `-ngl 999` (all
  layers on GPU) is the right default here.
