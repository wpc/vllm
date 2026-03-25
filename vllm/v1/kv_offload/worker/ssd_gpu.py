# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SSD KV cache offloading handlers.

Uses pinned CPU staging tensors (same layout as CpuGpuOffloadingHandlers)
with async io_uring for SSD persistence. The GPU<->CPU transfer uses
SingleDirectionOffloadingHandler from cpu_gpu.py. The CPU<->SSD transfer
happens in the background via the C++ _ssd_kv_C module.

Architecture:
  Store: GPU --(swap_blocks)--> staging CPU tensors
         staging CPU tensors --(io_uring async)--> SSD (background)
  Load:  SSD --(io_uring async)--> staging CPU tensors (background)
         staging CPU tensors --(swap_blocks)--> GPU
"""
import torch

from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.worker.cpu_gpu import (
    CpuGpuOffloadingHandlers,
)

logger = init_logger(__name__)


class SsdGpuOffloadingHandlers:
    """Creates GPU<->SSD transfer handlers using CPU staging tensors.

    Reuses CpuGpuOffloadingHandlers for the GPU<->CPU staging transfer,
    which handles all the tensor layout probing, swap_blocks, CUDA streams,
    and event-based completion tracking.

    SSD persistence is handled separately: after a GPU->CPU transfer completes,
    the staging tensor data is asynchronously flushed to SSD. Before a
    CPU->GPU transfer, the data is loaded from SSD into the staging tensor.

    For now, the SSD I/O is deferred — the staging CPU tensors serve as the
    primary offloading target (same as CPU offloading). SSD persistence will
    be added when io_uring integration is validated.
    """

    def __init__(
        self,
        gpu_block_size: int,
        ssd_block_size: int,
        num_ssd_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
        ssd_paths: list[str],
        block_bytes: int,
        page_size: int = 4096,
        io_queue_depth: int = 256,
        max_concurrent_transfers: int = 32,
        ram_cache_gb: float = 0,
    ):
        # Delegate to CpuGpuOffloadingHandlers which handles:
        # - GPU tensor layout probing (kernel_block_size, split_k_and_v, etc.)
        # - Allocating pinned CPU staging tensors (num_ssd_blocks in block dim)
        # - Creating SingleDirectionOffloadingHandler for GPU->CPU and CPU->GPU
        # - CUDA stream and event management
        self._cpu_gpu_handlers = CpuGpuOffloadingHandlers(
            gpu_block_size=gpu_block_size,
            cpu_block_size=ssd_block_size,
            num_cpu_blocks=num_ssd_blocks,
            gpu_caches=gpu_caches,
            attn_backends=attn_backends,
        )

        # Expose the handlers with the same interface
        self.gpu_to_ssd_handler = self._cpu_gpu_handlers.gpu_to_cpu_handler
        self.ssd_to_gpu_handler = self._cpu_gpu_handlers.cpu_to_gpu_handler

        # Initialize SSD store for background persistence (future)
        self._ssd_store = None
        self._init_ssd_store(
            ssd_paths, num_ssd_blocks, block_bytes, page_size, io_queue_depth
        )

        logger.info(
            "SSD offloading handlers initialized: %d staging blocks, "
            "%d bytes/block, io_uring=%s",
            num_ssd_blocks,
            block_bytes,
            "enabled" if self._ssd_store is not None else "disabled (CPU-only)",
        )

    def _init_ssd_store(
        self,
        ssd_paths: list[str],
        num_blocks: int,
        block_bytes: int,
        page_size: int,
        io_queue_depth: int,
    ) -> None:
        """Initialize the C++ SSD store for background persistence."""
        try:
            import vllm._ssd_kv_C  # noqa: F401

            ssd_ops = torch.ops._ssd_kv_C

            file_paths = [f"{path}_{i}.dat" for i, path in enumerate(ssd_paths)]

            self._ssd_store_handle = ssd_ops.create(
                file_paths,
                num_blocks,
                block_bytes,
                page_size,
                io_queue_depth,
            )
            self._ssd_store = ssd_ops

            logger.info(
                "SSD KV store initialized: %d blocks, %d bytes/block, "
                "%d files",
                num_blocks,
                block_bytes,
                len(file_paths),
            )
        except (ImportError, Exception) as e:
            logger.warning(
                "SSD store not available (%s). Using CPU-only staging. "
                "Install liburing-dev and rebuild vLLM for io_uring support.",
                e,
            )
