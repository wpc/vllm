# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiered GPU<->SSD transfer handlers with RAM block cache.

Store path: GPU -> RAM cache (fast memcpy) + async flush RAM -> SSD
Load path:  Check RAM cache -> hit: RAM -> GPU (skip SSD)
                             -> miss: SSD -> RAM -> GPU
"""
import enum
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.v1.kv_offload.backends.ram_cache import RAMBlockCache
from vllm.v1.kv_offload.mediums import BlockIDsLoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

logger = init_logger(__name__)


class Phase(enum.Enum):
    CUDA_TO_RAM = 1     # CUDA async GPU -> RAM cache
    RAM_TO_SSD = 2      # io_uring RAM cache -> SSD
    SSD_TO_RAM = 3      # io_uring SSD -> RAM cache
    RAM_TO_GPU = 4      # CUDA async RAM cache -> GPU
    DONE = 5


@dataclass
class TieredTransfer:
    job_id: int
    phase: Phase
    ram_slot: int
    ssd_block_ids: np.ndarray
    gpu_block_ids: np.ndarray
    stream: torch.cuda.Stream
    cuda_event: torch.Event
    num_bytes: int


class TieredGpuToSsdHandler(OffloadingHandler):
    """GPU -> RAM cache -> SSD handler.

    On transfer_async:
      1. Allocate RAM cache slot (may evict old entry)
      2. CUDA async copy GPU -> RAM cache slot
    On get_finished:
      1. Poll CUDA events; when done, submit io_uring write RAM -> SSD
      2. Poll io_uring completions
      3. Return completed transfers
    """

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        gpu_block_size_factor: int,
        ram_cache: RAMBlockCache,
        ssd_store,
        block_size_in_bytes: list[int],
    ):
        self.gpu_tensors = gpu_tensors
        self.gpu_block_size_factor = gpu_block_size_factor
        self.ram_cache = ram_cache
        self.ssd_store = ssd_store
        self.block_size_in_bytes = block_size_in_bytes
        self.total_block_size_in_bytes = sum(block_size_in_bytes)

        self._transfers: deque[TieredTransfer] = deque()
        self._stream_pool: list[torch.cuda.Stream] = []
        self._event_pool: list[torch.Event] = []

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        gpu_blocks = src_spec.block_ids
        ssd_blocks = dst_spec.block_ids

        # Allocate RAM cache slot for the first SSD block
        # (for simplicity, we cache the first block of the transfer)
        ssd_block_id = int(ssd_blocks[0]) if len(ssd_blocks) > 0 else 0
        ram_slot, evicted = self.ram_cache.put(ssd_block_id)

        stream = (
            self._stream_pool.pop()
            if self._stream_pool
            else torch.cuda.Stream()
        )
        cuda_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=False)
        )

        # CUDA async copy GPU -> RAM cache slot
        from vllm.v1.kv_offload.worker.cpu_gpu import expand_block_ids

        num_sub_blocks = len(gpu_blocks) * self.gpu_block_size_factor
        src_expanded = np.empty(num_sub_blocks, dtype=np.int64)
        expand_block_ids(gpu_blocks, self.gpu_block_size_factor, src_expanded)
        dst_sequential = np.arange(num_sub_blocks, dtype=np.int64)
        src_to_dst = np.stack([src_expanded, dst_sequential], axis=1)
        src_to_dst_tensor = torch.from_numpy(src_to_dst)

        ram_tensor = self.ram_cache.get_buffer_tensor(ram_slot)

        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            offset = 0
            for gpu_tensor, bsz in zip(
                self.gpu_tensors, self.block_size_in_bytes
            ):
                total_bytes = bsz * num_sub_blocks
                staging_view = ram_tensor[offset : offset + total_bytes]
                staging_reshaped = staging_view.view(
                    num_sub_blocks, *gpu_tensor.shape[1:]
                )
                ops.swap_blocks(
                    gpu_tensor, staging_reshaped, bsz, src_to_dst_tensor
                )
                offset += total_bytes
            cuda_event.record(stream)

        num_bytes = num_sub_blocks * self.total_block_size_in_bytes
        self._transfers.append(
            TieredTransfer(
                job_id=job_id,
                phase=Phase.CUDA_TO_RAM,
                ram_slot=ram_slot,
                ssd_block_ids=ssd_blocks,
                gpu_block_ids=gpu_blocks,
                stream=stream,
                cuda_event=cuda_event,
                num_bytes=num_bytes,
            )
        )
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []

        # Advance CUDA_TO_RAM -> RAM_TO_SSD
        for transfer in self._transfers:
            if transfer.phase == Phase.CUDA_TO_RAM:
                if transfer.cuda_event.query():
                    # CUDA copy done, submit io_uring write
                    buf_ptr = self.ram_cache.get_buffer_ptr(transfer.ram_slot)
                    for i, ssd_block_id in enumerate(transfer.ssd_block_ids):
                        self.ssd_store.write_block(
                            int(ssd_block_id),
                            buf_ptr,
                            transfer.job_id * 1000 + i,
                        )
                    transfer.phase = Phase.RAM_TO_SSD

        # Poll io_uring completions
        io_completions = self.ssd_store.poll()
        completed_jobs = {jid // 1000 for jid, _, _ in io_completions}

        for transfer in self._transfers:
            if (
                transfer.phase == Phase.RAM_TO_SSD
                and transfer.job_id in completed_jobs
            ):
                transfer.phase = Phase.DONE

        # Collect completed
        while self._transfers and self._transfers[0].phase == Phase.DONE:
            transfer = self._transfers.popleft()
            # Don't free RAM slot — keep it cached for future reads
            self._stream_pool.append(transfer.stream)
            self._event_pool.append(transfer.cuda_event)
            results.append(
                TransferResult(
                    job_id=transfer.job_id,
                    success=True,
                    transfer_size=transfer.num_bytes,
                    transfer_type=("GPU", "SSD"),
                )
            )
        return results

    def wait(self, job_ids: set[int]) -> None:
        for transfer in self._transfers:
            if transfer.job_id in job_ids:
                transfer.cuda_event.synchronize()
        self.ssd_store.wait_all()


class TieredSsdToGpuHandler(OffloadingHandler):
    """SSD -> RAM cache -> GPU handler with cache lookup.

    On transfer_async:
      1. Check RAM cache for the SSD block
      2. If hit: CUDA async copy RAM -> GPU directly
      3. If miss: submit io_uring read SSD -> RAM, then CUDA copy
    """

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        gpu_block_size_factor: int,
        ram_cache: RAMBlockCache,
        ssd_store,
        block_size_in_bytes: list[int],
    ):
        self.gpu_tensors = gpu_tensors
        self.gpu_block_size_factor = gpu_block_size_factor
        self.ram_cache = ram_cache
        self.ssd_store = ssd_store
        self.block_size_in_bytes = block_size_in_bytes
        self.total_block_size_in_bytes = sum(block_size_in_bytes)

        self._transfers: deque[TieredTransfer] = deque()
        self._stream_pool: list[torch.cuda.Stream] = []
        self._event_pool: list[torch.Event] = []
        self._cache_hits = 0
        self._cache_misses = 0

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        ssd_blocks = src_spec.block_ids
        gpu_blocks = dst_spec.block_ids

        stream = (
            self._stream_pool.pop()
            if self._stream_pool
            else torch.cuda.Stream()
        )
        cuda_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=False)
        )

        ssd_block_id = int(ssd_blocks[0]) if len(ssd_blocks) > 0 else 0
        ram_slot = self.ram_cache.get(ssd_block_id)

        num_sub_blocks = len(gpu_blocks) * self.gpu_block_size_factor
        num_bytes = num_sub_blocks * self.total_block_size_in_bytes

        if ram_slot is not None:
            # Cache hit — copy directly from RAM to GPU
            self._cache_hits += 1
            self._start_ram_to_gpu(
                ram_slot, gpu_blocks, num_sub_blocks, stream, cuda_event
            )
            self._transfers.append(
                TieredTransfer(
                    job_id=job_id,
                    phase=Phase.RAM_TO_GPU,
                    ram_slot=ram_slot,
                    ssd_block_ids=ssd_blocks,
                    gpu_block_ids=gpu_blocks,
                    stream=stream,
                    cuda_event=cuda_event,
                    num_bytes=num_bytes,
                )
            )
        else:
            # Cache miss — read from SSD first
            self._cache_misses += 1
            ram_slot, _evicted = self.ram_cache.put(ssd_block_id)

            buf_ptr = self.ram_cache.get_buffer_ptr(ram_slot)
            for i, ssd_bid in enumerate(ssd_blocks):
                self.ssd_store.read_block(
                    int(ssd_bid), buf_ptr, job_id * 1000 + i
                )

            self._transfers.append(
                TieredTransfer(
                    job_id=job_id,
                    phase=Phase.SSD_TO_RAM,
                    ram_slot=ram_slot,
                    ssd_block_ids=ssd_blocks,
                    gpu_block_ids=gpu_blocks,
                    stream=stream,
                    cuda_event=cuda_event,
                    num_bytes=num_bytes,
                )
            )

        if (self._cache_hits + self._cache_misses) % 100 == 0:
            total = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total if total > 0 else 0
            logger.info(
                "RAM cache stats: %d hits, %d misses, %.1f%% hit rate, %s",
                self._cache_hits,
                self._cache_misses,
                hit_rate * 100,
                self.ram_cache.hit_rate_str,
            )

        return True

    def _start_ram_to_gpu(
        self,
        ram_slot: int,
        gpu_blocks: np.ndarray,
        num_sub_blocks: int,
        stream: torch.cuda.Stream,
        cuda_event: torch.Event,
    ) -> None:
        from vllm.v1.kv_offload.worker.cpu_gpu import expand_block_ids

        ram_tensor = self.ram_cache.get_buffer_tensor(ram_slot)
        src_sequential = np.arange(num_sub_blocks, dtype=np.int64)
        dst_expanded = np.empty(num_sub_blocks, dtype=np.int64)
        expand_block_ids(gpu_blocks, self.gpu_block_size_factor, dst_expanded)
        src_to_dst = np.stack([src_sequential, dst_expanded], axis=1)
        src_to_dst_tensor = torch.from_numpy(src_to_dst)

        with torch.cuda.stream(stream):
            offset = 0
            for gpu_tensor, bsz in zip(
                self.gpu_tensors, self.block_size_in_bytes
            ):
                total_bytes = bsz * num_sub_blocks
                staging_view = ram_tensor[offset : offset + total_bytes]
                staging_reshaped = staging_view.view(
                    num_sub_blocks, *gpu_tensor.shape[1:]
                )
                ops.swap_blocks(
                    staging_reshaped, gpu_tensor, bsz, src_to_dst_tensor
                )
                offset += total_bytes
            cuda_event.record(stream)

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []

        # Poll io_uring completions for SSD_TO_RAM transfers
        io_completions = self.ssd_store.poll()
        completed_jobs = {jid // 1000 for jid, _, _ in io_completions}

        for transfer in self._transfers:
            if (
                transfer.phase == Phase.SSD_TO_RAM
                and transfer.job_id in completed_jobs
            ):
                # SSD read done — start CUDA copy RAM -> GPU
                num_sub_blocks = (
                    len(transfer.gpu_block_ids) * self.gpu_block_size_factor
                )
                self._start_ram_to_gpu(
                    transfer.ram_slot,
                    transfer.gpu_block_ids,
                    num_sub_blocks,
                    transfer.stream,
                    transfer.cuda_event,
                )
                transfer.phase = Phase.RAM_TO_GPU

        # Check CUDA event completions for RAM_TO_GPU transfers
        for transfer in self._transfers:
            if transfer.phase == Phase.RAM_TO_GPU:
                if transfer.cuda_event.query():
                    transfer.phase = Phase.DONE

        # Collect completed
        while self._transfers and self._transfers[0].phase == Phase.DONE:
            transfer = self._transfers.popleft()
            self._stream_pool.append(transfer.stream)
            self._event_pool.append(transfer.cuda_event)
            results.append(
                TransferResult(
                    job_id=transfer.job_id,
                    success=True,
                    transfer_size=transfer.num_bytes,
                    transfer_type=("SSD", "GPU"),
                )
            )
        return results

    def wait(self, job_ids: set[int]) -> None:
        self.ssd_store.wait_all()
        for transfer in self._transfers:
            if transfer.job_id in job_ids:
                transfer.cuda_event.synchronize()
