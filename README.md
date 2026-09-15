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
| `build.lock`  | Generated pins for `./build.sh --reproduce`. |

## Build

```bash
./build.sh                 # latest llama.cpp master HEAD
./build.sh --ref b10375    # a specific tag/branch/commit (current pin)
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
./run-muse-glimmer.sh                  # unsloth/Muse-Glimmer-30B-GGUF (UD-Q6_K_XL)
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
  weights and dies. `CTX` defaults to `32768`, and it is the **total** budget
  llama.cpp splits across slots — raising `PARALLEL` divides it rather than
  multiplying, because at ~95 GiB resident there is no memory to spare.
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
./run-deepseek-v4-flash.sh                  # IQ2_M + DSpark, 32k ctx
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
`build.lock` pins b10375, clear of that window.

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
`build.lock` pins b10375.

> **Memory**: the Spark's 121 GB of unified memory is shared with anything else
> running on the box. Another large model already resident will not leave room
> for this one — stop other GPU-heavy containers first (`docker ps`, then
> `docker stop <name>`).

### Useful env vars (see `run.sh` header)

`IMAGE`, `PORT` (8080), `GPU_LAYERS` (999=all), `HF_TOKEN`, `HF_HOME`, `DETACH`.

`-hf` downloads go into the standard Hugging Face cache: `HF_HOME` defaults to
`~/.cache/huggingface`, shared with any other HF tooling on the host. llama.cpp's
flat `-hf` cache lands in a `llama.cpp/` subdir of it — its layout differs from
the HF hub `models--org--repo` layout, so files aren't deduped across the two,
but all models stay under one directory.

## Notes

- **Current pin**: release tag **`b10375`** (`ba360efe1`). Pinning a release tag
  rather than tracking `master` keeps `build.lock` reproducible.
  DeepSeek-V4-Flash DSpark needs ≥ `b10269`.
- **Verified baseline**: bare-metal build `c1304d7b2 (9671)` ran Qwen3.6-35B-A3B
  Q8 on the GB10 at ~697 t/s prefill / ~48 t/s decode, all layers on CUDA — this
  image reproduces that build inside a container.
- The image build needs network for the embedded Web UI assets (Hugging Face
  bucket `ggml-org/llama-ui`). Use `--no-ui` for a hermetic, API-only image.
- Unified memory: GB10 shares 128 GB between CPU and GPU, so `-ngl 999` (all
  layers on GPU) is the right default here.
