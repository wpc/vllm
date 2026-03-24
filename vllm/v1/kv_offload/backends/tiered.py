# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import LoadStoreSpec
from vllm.v1.kv_offload.backend import Backend, BlockStatus
from vllm.v1.kv_offload.backends.ssd import SSDBackend

logger = init_logger(__name__)


class TieredSSDBackend(Backend):
    """Tiered backend: RAM cache -> SSD.

    Transparent to the OffloadingManager — it sees a single Backend.
    The RAM cache is managed internally by the transfer handlers
    (tiered_ssd_gpu.py), not by this backend class.

    This backend delegates all block management to the underlying
    SSD backend. The RAM cache is purely a transfer-level optimization
    that sits in the handlers, not the scheduler-side backend.
    """

    def __init__(self, ssd_backend: SSDBackend):
        super().__init__(
            block_size=ssd_backend.block_size,
            medium=ssd_backend.medium,
        )
        self._ssd_backend = ssd_backend

    def get_num_free_blocks(self):
        return self._ssd_backend.get_num_free_blocks()

    def allocate_blocks(self, block_hashes: list[BlockHash]) -> list[BlockStatus]:
        return self._ssd_backend.allocate_blocks(block_hashes)

    def free(self, block: BlockStatus):
        self._ssd_backend.free(block)

    def get_load_store_spec(
        self, block_hashes: Iterable[BlockHash], blocks: Iterable[BlockStatus]
    ) -> LoadStoreSpec:
        return self._ssd_backend.get_load_store_spec(block_hashes, blocks)
