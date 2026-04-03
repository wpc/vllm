ARG BASE_IMAGE=vllm/vllm-openai:nightly
ARG DOCA_VERSION=3.2.0
ARG DOCA_BUILD_ID=125000-25.10
ARG UCX_VERSION=1.20.0

#################### UCX BUILD STAGE ####################
# Build UCX v1.20.0 from source against DOCA 3.2.0 rdma-core
# (Data Direct DMA-BUF support for GPUDirect RDMA on GB300)
FROM ${BASE_IMAGE} AS ucx-build

ARG DOCA_VERSION
ARG DOCA_BUILD_ID
ARG UCX_VERSION

# Install DOCA repository for rdma-core with Data Direct DMA-BUF support
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends wget && \
    ARCH_SUFFIX=$(dpkg --print-architecture) && \
    MELLANOX_OS="$(. /etc/lsb-release; echo ${DISTRIB_ID}${DISTRIB_RELEASE} | tr A-Z a-z | tr -d .)" && \
    wget --tries=3 --waitretry=5 --no-verbose \
        "https://www.mellanox.com/downloads/DOCA/DOCA_v${DOCA_VERSION}/host/doca-host_${DOCA_VERSION}-${DOCA_BUILD_ID}-${MELLANOX_OS}_${ARCH_SUFFIX}.deb" \
        -O /tmp/doca-host.deb && \
    dpkg -i /tmp/doca-host.deb && \
    rm -f /tmp/doca-host.deb && \
    apt-get update

# Reinstall rdma-core from DOCA repo (force DOCA version with Data Direct)
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --reinstall \
        libibverbs-dev rdma-core ibverbs-utils libibumad-dev \
        libnuma-dev librdmacm-dev ibverbs-providers

# Verify Data Direct DMA-BUF support in DOCA rdma-core
RUN grep -q "mlx5dv_get_data_direct_sysfs_path" /usr/include/infiniband/mlx5dv.h \
    && echo "Data Direct header: OK"

# Install UCX build dependencies (cuda-nvml-dev for nvml.h)
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential pkg-config cuda-nvml-dev-13-0

# Build UCX from source with Data Direct detection
RUN wget --tries=3 --waitretry=5 --no-verbose \
        "https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz" \
        -O /tmp/ucx.tar.gz && \
    tar xzf /tmp/ucx.tar.gz -C /tmp && \
    rm -f /tmp/ucx.tar.gz && \
    cd /tmp/ucx-${UCX_VERSION} && \
    ./configure \
        --disable-logging \
        --disable-debug \
        --disable-assertions \
        --disable-params-check \
        --prefix=/usr/local/ucx \
        --enable-shared \
        --disable-static \
        --disable-doxygen-doc \
        --enable-optimizations \
        --enable-cma \
        --enable-devel-headers \
        --with-cuda=/usr/local/cuda \
        --with-verbs \
        --with-dm \
        --enable-mt && \
    grep "HAVE_DECL_MLX5DV_GET_DATA_DIRECT_SYSFS_PATH.*1" config.h && \
    echo "UCX configure: Data Direct detected" && \
    make -j$(nproc) && \
    make install

#################### FINAL IMAGE ####################
FROM ${BASE_IMAGE}
ARG CUDA_MAJOR_VERSION=12
ARG DOCA_VERSION=3.2.0
ARG DOCA_BUILD_ID=125000-25.10

# Install DOCA rdma-core runtime libraries
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends wget && \
    ARCH_SUFFIX=$(dpkg --print-architecture) && \
    MELLANOX_OS="$(. /etc/lsb-release; echo ${DISTRIB_ID}${DISTRIB_RELEASE} | tr A-Z a-z | tr -d .)" && \
    wget --tries=3 --waitretry=5 --no-verbose \
        "https://www.mellanox.com/downloads/DOCA/DOCA_v${DOCA_VERSION}/host/doca-host_${DOCA_VERSION}-${DOCA_BUILD_ID}-${MELLANOX_OS}_${ARCH_SUFFIX}.deb" \
        -O /tmp/doca-host.deb && \
    dpkg -i /tmp/doca-host.deb && \
    rm -f /tmp/doca-host.deb && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --reinstall \
        rdma-core libibverbs-dev librdmacm-dev ibverbs-providers ibverbs-utils && \
    rm -rf /var/lib/apt/lists/*

# Copy pre-built UCX from build stage
COPY --from=ucx-build /usr/local/ucx/ /usr/local/ucx/

# UCX environment
ENV PATH="/usr/local/ucx/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/ucx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

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

# Replace NIXL's bundled UCX with our GDR-enabled UCX.
# NIXL's pip package vendors its own UCX via rpath (nixl_cu13.libs/).
# The bundled UCX lacks Data Direct DMA-BUF support, and ELF loader
# follows rpath before LD_LIBRARY_PATH. Must replace both core libs
# AND UCX plugins to avoid ABI mismatches.
# NOTE: This MUST run after all pip installs that touch nixl.
RUN NIXL_LIBS=$(python3 -c "import pathlib; import nixl; p=pathlib.Path(nixl.__file__).parent.parent/'nixl_cu13.libs'; print(p)") && \
    echo "NIXL libs dir: $NIXL_LIBS" && \
    replaced=0 && \
    echo "--- Replacing core UCX libs ---" && \
    for lib in libucp libuct libucs libucm; do \
        bundled=$(find "${NIXL_LIBS}" -maxdepth 1 -name "${lib}-*.so*" -print -quit) ; \
        if [ -n "$bundled" ]; then \
            ln -sf /usr/local/ucx/lib/${lib}.so.0 "$bundled" && \
            echo "Replaced $(basename $bundled) -> /usr/local/ucx/lib/${lib}.so.0" && \
            replaced=$((replaced + 1)) ; \
        fi ; \
    done && \
    echo "--- Replacing UCX plugins ---" && \
    if [ -d "${NIXL_LIBS}/ucx" ]; then \
        rm -rf "${NIXL_LIBS}/ucx" && \
        ln -sf /usr/local/ucx/lib/ucx "${NIXL_LIBS}/ucx" && \
        echo "Replaced plugins dir -> /usr/local/ucx/lib/ucx" && \
        replaced=$((replaced + 1)) ; \
    fi && \
    if [ "$replaced" -eq 0 ]; then \
        echo "ERROR: No NIXL bundled UCX libs found — overlay is pointless" && exit 1 ; \
    fi && \
    echo "Replaced $replaced UCX components"

# Verify UCX installation and NIXL linkage
RUN ucx_info -v && \
    python3 -c "import nixl; print('NIXL import OK')" && \
    echo "UCX with RDMA + NIXL linkage verified"
