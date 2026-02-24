ARG BASE_IMAGE=vllm/vllm-openai:nightly
FROM ${BASE_IMAGE}
ARG CUDA_MAJOR_VERSION=12

# Install nsys CLI
RUN curl -L -O https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_6/nsight-systems-cli-2025.6.1_2025.6.1.190-1_arm64.deb \
    && sudo dpkg -i nsight-systems-cli-2025.6.1_2025.6.1.190-1_arm64.deb \
    && rm nsight-systems-cli-2025.6.1_2025.6.1.190-1_arm64.deb

RUN curl -L -O https://github.com/casey/just/releases/download/1.43.1/just-1.43.1-aarch64-unknown-linux-musl.tar.gz \
    && tar -xzf just-1.43.1-aarch64-unknown-linux-musl.tar.gz --directory /usr/local/bin \
    && rm -rf just-1.43.1-aarch64-unknown-linux-musl.tar.gz just-1.43.1-aarch64-unknown-linux-musl

RUN uv pip uninstall --system flashinfer_python flashinfer-cubin flashinfer-jit-cache
RUN uv pip install --system --no-deps --no-build-isolation flashinfer_python==v0.6.0rc2 \
    https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.0rc2/flashinfer_cubin-0.6.0rc2-py3-none-any.whl \
    https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.0rc2/flashinfer_jit_cache-0.6.0rc2+cu130-cp39-abi3-manylinux_2_28_aarch64.whl


RUN uv pip uninstall --system nvidia-nvshmem-cu13
RUN uv pip uninstall --system nvidia-nvshmem-cu12
RUN uv pip install --system --no-deps nvidia-nvshmem-cu${CUDA_MAJOR_VERSION}==3.4.5
RUN uv pip install --system nixl[cu${CUDA_MAJOR_VERSION}] pandas datasets nvtx lm_eval[api]

RUN sudo apt-get update -qq && sudo apt-get install -y -qq pkg-config libssl-dev protobuf-compiler git

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN git clone -b queue_size https://github.com/minosfuture/router.git --depth 1 && cd router && cargo build --release
ENV TRITON_PTXAS_PATH /usr/local/cuda/bin/ptxas

RUN pip install --upgrade pip && pip install \
    "opentelemetry-sdk>=1.26.0,<1.27.0" \
    "opentelemetry-api>=1.26.0,<1.27.0" \
    "opentelemetry-exporter-otlp>=1.26.0,<1.27.0" \
    "opentelemetry-semantic-conventions-ai>=0.4.1,<0.5.0" \
    "opentelemetry-instrumentation" \
    "opentelemetry-instrumentation-fastapi" \
    "opentelemetry-instrumentation-starlette"
