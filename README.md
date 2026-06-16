# llama.cpp on DGX Spark (GB10) — Dockerized server

From-source, reproducible Docker build of the llama.cpp **server** for the
NVIDIA DGX Spark (GB10 Grace Blackwell, `sm_121a`), built with CUDA 13.

Same shape as the sibling `~/vllm/` setup: pin a known-good CUDA base image,
recompile the GPU kernels for this exact chip, and record the pins in
`build.lock` for reproducible rebuilds.

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
| `run.sh`      | Serve a HF repo or a local GGUF with `--gpus all`. |
| `build.lock`  | Generated pins for `./build.sh --reproduce`. |

## Build

```bash
./build.sh                 # latest llama.cpp master HEAD
./build.sh --ref b9671     # a specific tag/branch/commit
./build.sh --reproduce     # rebuild exactly what build.lock records
./build.sh --no-ui         # skip the embedded Web UI (no build-time HF fetch)
```

First build compiles the CUDA kernels (~several minutes on the Spark); `ccache`
is mounted as a BuildKit cache so rebuilds are fast.

## Serve

```bash
# Pull + serve a GGUF straight from Hugging Face (cached under ~/.cache/llama.cpp)
./run.sh ggml-org/gemma-3-4b-it-GGUF

# Serve a local GGUF (its directory is mounted read-only)
./run.sh /home/jesse/Development/models/<model>.gguf

# Extra llama-server flags pass straight through
./run.sh ggml-org/gemma-3-4b-it-GGUF --ctx-size 32768 --parallel 4

# Background server mode (auto-restart)
DETACH=1 ./run.sh ggml-org/gemma-3-4b-it-GGUF
```

Then: OpenAI-compatible API at `http://localhost:8080/v1/chat/completions`, and
the Web UI at `http://localhost:8080/`.

### Useful env vars (see `run.sh` header)

`IMAGE`, `PORT` (8080), `GPU_LAYERS` (999=all), `HF_TOKEN`, `LLAMA_CACHE`, `DETACH`.

## Notes

- **Verified baseline**: bare-metal build `c1304d7b2 (9671)` ran Qwen3.6-35B-A3B
  Q8 on the GB10 at ~697 t/s prefill / ~48 t/s decode, all layers on CUDA — this
  image reproduces that build inside a container.
- The image build needs network for the embedded Web UI assets (Hugging Face
  bucket `ggml-org/llama-ui`). Use `--no-ui` for a hermetic, API-only image.
- Unified memory: GB10 shares 128 GB between CPU and GPU, so `-ngl 999` (all
  layers on GPU) is the right default here.
