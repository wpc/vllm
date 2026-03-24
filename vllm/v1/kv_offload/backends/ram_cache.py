# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict

import torch

from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available

logger = init_logger(__name__)


class RAMBlockCache:
    """In-memory block-level LRU cache sitting between GPU and SSD.

    Pre-allocates contiguous pinned CPU memory for all cache slots.
    Uses OrderedDict for O(1) LRU lookup and eviction.

    Inspired by embedding_db's ShardedLruRowCache but simplified
    for block-level (not row-level) caching without sharding.
    """

    def __init__(self, num_slots: int, bytes_per_slot: int):
        self.num_slots = num_slots
        self.bytes_per_slot = bytes_per_slot

        pin_memory = is_pin_memory_available()
        # Pre-allocate contiguous pinned memory for all cache slots
        self.buffer = torch.empty(
            num_slots,
            bytes_per_slot,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=pin_memory,
        )

        # LRU tracking: ssd_block_id -> slot_index
        # OrderedDict maintains insertion order; move_to_end for MRU
        self._lru: OrderedDict[int, int] = OrderedDict()

        # Free list of slot indices
        self._free_slots: list[int] = list(range(num_slots))

        logger.info(
            "RAM block cache initialized: %d slots, %d bytes/slot, "
            "%.1f MB total",
            num_slots,
            bytes_per_slot,
            num_slots * bytes_per_slot / (1024**2),
        )

    def get(self, ssd_block_id: int) -> int | None:
        """Look up a block in the cache.

        Returns the slot index if cached (and moves to MRU position),
        or None if not cached.
        """
        slot = self._lru.get(ssd_block_id)
        if slot is not None:
            self._lru.move_to_end(ssd_block_id)
        return slot

    def put(self, ssd_block_id: int) -> tuple[int, int | None]:
        """Allocate a cache slot for a block.

        Returns (slot_index, evicted_ssd_block_id or None).
        If the block is already cached, returns its existing slot.
        """
        # Already cached — just promote to MRU
        existing = self._lru.get(ssd_block_id)
        if existing is not None:
            self._lru.move_to_end(ssd_block_id)
            return existing, None

        evicted_id: int | None = None

        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            # Evict LRU entry (first item in OrderedDict)
            evicted_id, slot = self._lru.popitem(last=False)

        self._lru[ssd_block_id] = slot
        return slot, evicted_id

    def remove(self, ssd_block_id: int) -> None:
        """Remove a block from the cache (e.g., on SSD block free)."""
        slot = self._lru.pop(ssd_block_id, None)
        if slot is not None:
            self._free_slots.append(slot)

    def get_buffer_ptr(self, slot: int) -> int:
        """Get raw data pointer for a cache slot."""
        return self.buffer[slot].data_ptr()

    def get_buffer_tensor(self, slot: int) -> torch.Tensor:
        """Get the buffer tensor for a cache slot."""
        return self.buffer[slot]

    @property
    def size(self) -> int:
        """Number of blocks currently cached."""
        return len(self._lru)

    @property
    def capacity(self) -> int:
        return self.num_slots

    @property
    def hit_rate_str(self) -> str:
        return f"{self.size}/{self.num_slots} slots used"
