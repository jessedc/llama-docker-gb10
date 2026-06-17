# syntax=docker/dockerfile:1.7
#
# Build llama.cpp (server) from source for the NVIDIA DGX Spark
# (GB10 Grace Blackwell, sm_121a).
#
# Strategy: a two-stage build.
#   1. BUILD stage  FROM nvidia/cuda:<ver>-devel  -- has nvcc + the full CUDA 13
#      toolkit. We clone a pinned upstream llama.cpp revision and compile its
#      CUDA kernels for compute_121a (the GB10 GPU), so we get real Blackwell
#      kernels instead of a PTX-JIT compatibility fallback.
#   2. RUNTIME stage FROM nvidia/cuda:<ver>-runtime -- slim image with only the
#      CUDA runtime libs (cudart/cublas/...). We copy just the built binary and
#      its shared libs. The real GPU driver (libcuda.so.1) is injected at run
#      time by the NVIDIA container runtime via `--gpus all`.
#
# Why build instead of `docker pull`: the official ghcr.io/ggml-org/llama.cpp
# CUDA images default to CUDA 12 (no sm_121 support at all) and the -cuda13
# variants are not GPU-CI-tested nor tuned for sm_121a. Building here is the
# same "pin a known-good CUDA base, recompile kernels for sm_121a" approach
# used for the vLLM Spark image.

ARG CUDA_VERSION=13.0.3
ARG UBUNTU_VERSION=ubuntu24.04

# ---------------------------------------------------------------------------
# Stage 1: build
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU_VERSION} AS build

# --- build configuration (override via --build-arg / build.sh) -------------
ARG LLAMA_REPO=https://github.com/ggml-org/llama.cpp.git
ARG LLAMA_REF=master
# GB10 Grace Blackwell compute capability. "121a" -> sm_121a (Blackwell-specific
# kernels). ggml's CMake rewrites a plain "121" to "121a" anyway.
ARG CUDA_ARCH=121a
# Build parallelism (Spark has 20 cores / 128 GB unified RAM).
ARG MAX_JOBS=16
# Embed the server Web UI. Needs network at build time to fetch the prebuilt
# UI assets from Hugging Face (the bucket MUST include the org prefix, else the
# download 404s -- this was a real footgun on a stale local cache).
ARG LLAMA_BUILD_UI=ON

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_BUILD_TYPE=Release \
    CCACHE_DIR=/ccache
# The CUDA driver lib is absent at build time (no GPU in the builder); the devel
# image ships a stub libcuda.so under .../stubs purely so the link step resolves
# cuda_driver. Putting it on LIBRARY_PATH guarantees the linker finds it.
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git cmake ccache build-essential curl ca-certificates \
      libcurl4-openssl-dev libssl-dev libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Fetch the requested llama.cpp revision (branch, tag, or commit SHA). An
# explicit fetch of the ref keeps this robust when master advances between
# builds and the cached clone layer predates the requested commit.
WORKDIR /opt
RUN git clone --filter=blob:none ${LLAMA_REPO} llama.cpp
WORKDIR /opt/llama.cpp
RUN git fetch --depth 1 origin ${LLAMA_REF} \
 && git checkout FETCH_HEAD \
 && git rev-parse HEAD | tee /opt/llama-commit.txt

# Configure + build only the server target (and its deps). ccache is mounted as
# a BuildKit cache so repeat builds reuse object files.
#  -DGGML_NATIVE=ON       : tune the CPU fallback for this exact chip (the build
#                           runs on the Spark itself -> armv9.2-a+sve2+i8mm+bf16).
#  -DCMAKE_CUDA_ARCHITECTURES=121a : GB10 Blackwell kernels.
#  -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined :
#      The build stage has only the CUDA *stub* libcuda.so, not the real driver,
#      so libggml-cuda.so carries undefined driver-API symbols (cuMemMap, etc.).
#      Allow them at link time; they resolve at runtime against the driver the
#      NVIDIA container runtime injects via `--gpus all`. (Same flag the official
#      llama.cpp .devops/cuda.Dockerfile uses.)
#  -DLLAMA_CURL=ON        : let llama-server pull GGUFs via -hf at runtime.
#  -DLLAMA_OPENSSL=ON     : compile in HTTPS support for -hf downloads. Newer
#                           llama.cpp gates its HTTP client's TLS behind an SSL
#                           backend; without OpenSSL dev files at configure time
#                           HTTPS is silently disabled and every -hf pull dies
#                           with "HTTPS is not supported. Please rebuild ...".
RUN --mount=type=cache,target=/ccache \
    cmake -B build \
      -DGGML_CUDA=ON \
      -DGGML_NATIVE=ON \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
      -DLLAMA_CURL=ON \
      -DLLAMA_OPENSSL=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_TOOLS=ON \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=ON \
      -DLLAMA_BUILD_UI=${LLAMA_BUILD_UI} \
      -DLLAMA_USE_PREBUILT_UI=ON \
      -DLLAMA_UI_HF_BUCKET=ggml-org/llama-ui \
 && cmake --build build --config Release -j ${MAX_JOBS} --target llama-server

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libcurl4 libssl3 libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy the server binary plus every ggml/llama shared lib it links. They all
# live together in build/bin, so copy that directory and point the loader at it.
COPY --from=build /opt/llama.cpp/build/bin/ /app/
# Provenance: which upstream commit this image was built from.
COPY --from=build /opt/llama-commit.txt /etc/llama-source-commit

ENV LD_LIBRARY_PATH=/app:${LD_LIBRARY_PATH} \
    LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8080
EXPOSE 8080

ENTRYPOINT ["/app/llama-server"]
# Default: bind all interfaces on 8080. run.sh appends the model + extra flags.
CMD ["--host", "0.0.0.0", "--port", "8080"]
