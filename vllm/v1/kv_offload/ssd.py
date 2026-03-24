# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager
from vllm.v1.kv_offload.backends.ssd import SSDBackend
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec, SSDLoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.ssd_gpu import SsdGpuOffloadingHandlers
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

logger = init_logger(__name__)


class SSDOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        # SSD configuration from kv_connector_extra_config
        self.ssd_paths: list[str] = self.extra_config.get(
            "ssd_paths", ["/tmp/ssd_kv_test/vllm_kv"]
        )
        ssd_capacity_gb = self.extra_config.get("ssd_capacity_gb")
        if not ssd_capacity_gb:
            raise ValueError(
                "ssd_capacity_gb must be specified in kv_connector_extra_config"
            )
        self.ssd_capacity_bytes = int(float(ssd_capacity_gb) * (1024**3))

        self.io_queue_depth: int = int(
            self.extra_config.get("io_queue_depth", 256)
        )
        self.max_concurrent_transfers: int = int(
            self.extra_config.get("max_concurrent_transfers", 32)
        )
        self.ram_cache_gb: float = float(
            self.extra_config.get("ram_cache_gb", 0)
        )
        self.page_size: int = int(self.extra_config.get("page_size", 4096))

        # Calculate kv_bytes_per_offloaded_block (same as CPUOffloadingSpec)
        assert kv_cache_config is not None
        page_sizes = {
            kv_cache_group.kv_cache_spec.page_size_bytes
            for kv_cache_group in kv_cache_config.kv_cache_groups
        }
        assert len(page_sizes) == 1
        page_size_bytes = page_sizes.pop()
        kv_bytes_per_block = (
            page_size_bytes
            * len(kv_cache_config.kv_cache_tensors)
            * vllm_config.parallel_config.world_size
        )
        self.kv_bytes_per_offloaded_block = kv_bytes_per_block * (
            self.offloaded_block_size // self.gpu_block_size
        )

        # Align block bytes to page_size for O_DIRECT
        if self.kv_bytes_per_offloaded_block % self.page_size != 0:
            self.kv_bytes_per_offloaded_block = (
                (self.kv_bytes_per_offloaded_block + self.page_size - 1)
                // self.page_size
                * self.page_size
            )

        self.num_blocks = (
            self.ssd_capacity_bytes // self.kv_bytes_per_offloaded_block
            if self.kv_bytes_per_offloaded_block > 0
            else 0
        )

        logger.info(
            "SSD offloading: %d blocks, %d bytes/block, "
            "%.1f GB total, %d SSD files, RAM cache: %.1f GB",
            self.num_blocks,
            self.kv_bytes_per_offloaded_block,
            self.ssd_capacity_bytes / (1024**3),
            len(self.ssd_paths),
            self.ram_cache_gb,
        )

        self._manager: OffloadingManager | None = None
        self._handlers: SsdGpuOffloadingHandlers | None = None
        self.eviction_policy: str = self.extra_config.get(
            "eviction_policy", "lru"
        )

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None
                and kv_events_config.enable_kv_cache_events
            )

            backend = SSDBackend(
                block_size=self.offloaded_block_size,
                num_blocks=self.num_blocks,
            )

            if self.eviction_policy == "lru":
                self._manager = LRUOffloadingManager(
                    backend=backend, enable_events=enable_events
                )
            elif self.eviction_policy == "arc":
                self._manager = ARCOffloadingManager(
                    backend=backend, enable_events=enable_events
                )
            else:
                raise ValueError(
                    f"Unknown eviction policy: {self.eviction_policy}. "
                    f"Supported policies: lru, arc"
                )
        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[
        tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]
    ]:
        if not self._handlers:
            if not current_platform.is_cuda_alike():
                raise RuntimeError(
                    "SSD Offloading is currently only supported on "
                    "CUDA-alike GPUs"
                )

            self._handlers = SsdGpuOffloadingHandlers(
                attn_backends=attn_backends,
                gpu_block_size=self.gpu_block_size,
                ssd_block_size=self.offloaded_block_size,
                num_ssd_blocks=self.num_blocks,
                gpu_caches=kv_caches,
                ssd_paths=self.ssd_paths,
                block_bytes=self.kv_bytes_per_offloaded_block,
                page_size=self.page_size,
                io_queue_depth=self.io_queue_depth,
                max_concurrent_transfers=self.max_concurrent_transfers,
                ram_cache_gb=self.ram_cache_gb,
            )

        assert self._handlers is not None
        yield (
            GPULoadStoreSpec,
            SSDLoadStoreSpec,
            self._handlers.gpu_to_ssd_handler,
        )
        yield (
            SSDLoadStoreSpec,
            GPULoadStoreSpec,
            self._handlers.ssd_to_gpu_handler,
        )
