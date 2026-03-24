# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.mediums import BlockIDsLoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

logger = init_logger(__name__)


class TransferPhase(enum.Enum):
    CUDA_COPY = 1
    IO_PENDING = 2
    IO_SUBMITTED = 3
    DONE = 4


@dataclass
class InFlightTransfer:
    job_id: int
    phase: TransferPhase
    staging_slot: int  # index into the staging buffer pool
    ssd_block_ids: np.ndarray  # SSD block slot IDs
    gpu_block_ids: np.ndarray  # GPU block IDs
    stream: torch.cuda.Stream
    cuda_event: torch.Event
    num_bytes: int
    start_time: float | None = None


class StagingBufferPool:
    """Pool of pinned CPU staging buffers for GPU<->SSD transfers.

    Each slot is a contiguous pinned CPU buffer large enough to hold
    all KV data for one offloaded block across all layers.
    """

    def __init__(
        self,
        num_slots: int,
        bytes_per_slot: int,
        gpu_tensors: list[torch.Tensor],
        block_size_in_bytes: list[int],
    ):
        self.num_slots = num_slots
        self.bytes_per_slot = bytes_per_slot
        self.gpu_tensors = gpu_tensors
        self.block_size_in_bytes = block_size_in_bytes

        pin_memory = is_pin_memory_available()
        # Allocate contiguous pinned buffer for all slots
        self.buffer = torch.empty(
            num_slots,
            bytes_per_slot,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=pin_memory,
        )

        self._free_slots: list[int] = list(range(num_slots))

    def allocate(self) -> int | None:
        if not self._free_slots:
            return None
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)

    @property
    def available(self) -> int:
        return len(self._free_slots)

    def get_buffer_ptr(self, slot: int) -> int:
        """Get raw data pointer for a staging slot (for io_uring)."""
        return self.buffer[slot].data_ptr()

    def get_buffer_tensor(self, slot: int) -> torch.Tensor:
        """Get the staging buffer tensor for a slot."""
        return self.buffer[slot]


class GpuToSsdHandler(OffloadingHandler):
    """Handles GPU -> SSD transfers.

    Pipeline: GPU --(CUDA async)--> staging buffer --(io_uring)--> SSD

    State machine per transfer:
      CUDA_COPY: CUDA async copy GPU -> staging (in progress)
      IO_PENDING: CUDA copy done, io_uring write not yet submitted
      IO_SUBMITTED: io_uring write submitted, waiting for completion
      DONE: transfer complete
    """

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        gpu_block_size_factor: int,
        staging_pool: StagingBufferPool,
        ssd_store,
        block_size_in_bytes: list[int],
    ):
        self.gpu_tensors = gpu_tensors
        self.gpu_block_size_factor = gpu_block_size_factor
        self.staging_pool = staging_pool
        self.ssd_store = ssd_store
        self.block_size_in_bytes = block_size_in_bytes
        self.total_block_size_in_bytes = sum(block_size_in_bytes)

        self._transfers: deque[InFlightTransfer] = deque()
        self._stream_pool: list[torch.cuda.Stream] = []
        self._event_pool: list[torch.Event] = []

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        gpu_blocks = src_spec.block_ids
        ssd_blocks = dst_spec.block_ids

        staging_slot = self.staging_pool.allocate()
        if staging_slot is None:
            logger.warning(
                "No staging buffer available for GPU->SSD transfer %d",
                job_id,
            )
            return False

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

        # Copy GPU blocks to staging buffer using swap_blocks
        # The staging buffer is treated as a "CPU tensor" with 1 block
        staging_tensor = self.staging_pool.get_buffer_tensor(staging_slot)

        # Wait for model computation to finish before offloading
        stream.wait_stream(torch.cuda.current_stream())

        # Expand GPU block IDs for sub-block copies
        from vllm.v1.kv_offload.worker.cpu_gpu import expand_block_ids

        num_sub_blocks = len(gpu_blocks) * self.gpu_block_size_factor
        src_expanded = np.empty(num_sub_blocks, dtype=np.int64)
        expand_block_ids(
            gpu_blocks, self.gpu_block_size_factor, src_expanded
        )

        # Create contiguous dst block IDs (0, 1, 2, ...)
        # relative to the staging buffer layout
        dst_sequential = np.arange(num_sub_blocks, dtype=np.int64)
        src_to_dst = np.stack([src_expanded, dst_sequential], axis=1)
        src_to_dst_tensor = torch.from_numpy(src_to_dst)

        with torch.cuda.stream(stream):
            # Reshape staging tensor to match the expected layout for
            # each GPU tensor's data
            offset = 0
            for gpu_tensor, bsz in zip(
                self.gpu_tensors, self.block_size_in_bytes
            ):
                total_bytes = bsz * num_sub_blocks
                staging_view = staging_tensor[offset : offset + total_bytes]
                staging_view = staging_view.view(
                    num_sub_blocks, bsz // gpu_tensor.element_size()
                )
                # Use swap_blocks to copy GPU -> staging
                # staging_view needs to match GPU tensor shape at block level
                staging_reshaped = staging_view.view(
                    num_sub_blocks, *gpu_tensor.shape[1:]
                )
                ops.swap_blocks(
                    gpu_tensor,
                    staging_reshaped,
                    bsz,
                    src_to_dst_tensor,
                )
                offset += total_bytes

            cuda_event.record(stream)

        num_bytes = num_sub_blocks * self.total_block_size_in_bytes

        self._transfers.append(
            InFlightTransfer(
                job_id=job_id,
                phase=TransferPhase.CUDA_COPY,
                staging_slot=staging_slot,
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

        # First pass: advance state machines
        for transfer in self._transfers:
            if transfer.phase == TransferPhase.CUDA_COPY:
                if transfer.cuda_event.query():
                    # CUDA copy done, submit io_uring writes
                    buf_ptr = self.staging_pool.get_buffer_ptr(
                        transfer.staging_slot
                    )
                    for i, ssd_block_id in enumerate(transfer.ssd_block_ids):
                        block_ptr = buf_ptr + i * self.staging_pool.bytes_per_slot // max(len(transfer.ssd_block_ids), 1)
                        self.ssd_store.write_block(
                            int(ssd_block_id),
                            block_ptr,
                            transfer.job_id * 1000 + i,
                        )
                    transfer.phase = TransferPhase.IO_SUBMITTED

        # Poll io_uring completions
        if self.ssd_store is not None:
            io_completions = self.ssd_store.poll()
            completed_jobs = set()
            for job_id, success, _is_read in io_completions:
                parent_job = job_id // 1000
                completed_jobs.add(parent_job)

            # Check if all IO for a transfer is done
            for transfer in self._transfers:
                if (
                    transfer.phase == TransferPhase.IO_SUBMITTED
                    and transfer.job_id in completed_jobs
                ):
                    transfer.phase = TransferPhase.DONE

        # Collect completed transfers from the front of the queue
        while self._transfers and self._transfers[0].phase == TransferPhase.DONE:
            transfer = self._transfers.popleft()
            self.staging_pool.free(transfer.staging_slot)
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
        if self.ssd_store is not None:
            self.ssd_store.wait_all()


class SsdToGpuHandler(OffloadingHandler):
    """Handles SSD -> GPU transfers.

    Pipeline: SSD --(io_uring)--> staging buffer --(CUDA async)--> GPU

    State machine per transfer:
      IO_SUBMITTED: io_uring read submitted, waiting for completion
      CUDA_COPY: io_uring done, CUDA async copy staging -> GPU (in progress)
      DONE: transfer complete
    """

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        gpu_block_size_factor: int,
        staging_pool: StagingBufferPool,
        ssd_store,
        block_size_in_bytes: list[int],
    ):
        self.gpu_tensors = gpu_tensors
        self.gpu_block_size_factor = gpu_block_size_factor
        self.staging_pool = staging_pool
        self.ssd_store = ssd_store
        self.block_size_in_bytes = block_size_in_bytes
        self.total_block_size_in_bytes = sum(block_size_in_bytes)

        self._transfers: deque[InFlightTransfer] = deque()
        self._stream_pool: list[torch.cuda.Stream] = []
        self._event_pool: list[torch.Event] = []

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        ssd_blocks = src_spec.block_ids
        gpu_blocks = dst_spec.block_ids

        staging_slot = self.staging_pool.allocate()
        if staging_slot is None:
            logger.warning(
                "No staging buffer available for SSD->GPU transfer %d",
                job_id,
            )
            return False

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

        # Submit io_uring reads for SSD blocks into staging buffer
        buf_ptr = self.staging_pool.get_buffer_ptr(staging_slot)
        for i, ssd_block_id in enumerate(ssd_blocks):
            block_ptr = buf_ptr + i * self.staging_pool.bytes_per_slot // max(len(ssd_blocks), 1)
            self.ssd_store.read_block(
                int(ssd_block_id),
                block_ptr,
                job_id * 1000 + i,
            )

        num_sub_blocks = len(gpu_blocks) * self.gpu_block_size_factor
        num_bytes = num_sub_blocks * self.total_block_size_in_bytes

        self._transfers.append(
            InFlightTransfer(
                job_id=job_id,
                phase=TransferPhase.IO_SUBMITTED,
                staging_slot=staging_slot,
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

        # Poll io_uring completions
        if self.ssd_store is not None:
            io_completions = self.ssd_store.poll()
            completed_jobs = set()
            for job_id, success, _is_read in io_completions:
                parent_job = job_id // 1000
                completed_jobs.add(parent_job)

            # Advance IO_SUBMITTED -> CUDA_COPY for completed reads
            for transfer in self._transfers:
                if (
                    transfer.phase == TransferPhase.IO_SUBMITTED
                    and transfer.job_id in completed_jobs
                ):
                    # IO done, start CUDA copy staging -> GPU
                    self._start_cuda_copy(transfer)
                    transfer.phase = TransferPhase.CUDA_COPY

        # Check CUDA event completions
        for transfer in self._transfers:
            if transfer.phase == TransferPhase.CUDA_COPY:
                if transfer.cuda_event.query():
                    transfer.phase = TransferPhase.DONE

        # Collect completed transfers
        while self._transfers and self._transfers[0].phase == TransferPhase.DONE:
            transfer = self._transfers.popleft()
            self.staging_pool.free(transfer.staging_slot)
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

    def _start_cuda_copy(self, transfer: InFlightTransfer) -> None:
        """Start CUDA async copy from staging buffer to GPU."""
        from vllm.v1.kv_offload.worker.cpu_gpu import expand_block_ids

        gpu_blocks = transfer.gpu_block_ids
        num_sub_blocks = len(gpu_blocks) * self.gpu_block_size_factor
        staging_tensor = self.staging_pool.get_buffer_tensor(
            transfer.staging_slot
        )

        # Source: sequential blocks in staging (0, 1, 2, ...)
        src_sequential = np.arange(num_sub_blocks, dtype=np.int64)

        # Destination: expanded GPU block IDs
        dst_expanded = np.empty(num_sub_blocks, dtype=np.int64)
        expand_block_ids(
            gpu_blocks, self.gpu_block_size_factor, dst_expanded
        )

        src_to_dst = np.stack([src_sequential, dst_expanded], axis=1)
        src_to_dst_tensor = torch.from_numpy(src_to_dst)

        stream = transfer.stream
        with torch.cuda.stream(stream):
            offset = 0
            for gpu_tensor, bsz in zip(
                self.gpu_tensors, self.block_size_in_bytes
            ):
                total_bytes = bsz * num_sub_blocks
                staging_view = staging_tensor[offset : offset + total_bytes]
                staging_reshaped = staging_view.view(
                    num_sub_blocks, *gpu_tensor.shape[1:]
                )
                ops.swap_blocks(
                    staging_reshaped,
                    gpu_tensor,
                    bsz,
                    src_to_dst_tensor,
                )
                offset += total_bytes

            transfer.cuda_event.record(stream)

    def wait(self, job_ids: set[int]) -> None:
        if self.ssd_store is not None:
            self.ssd_store.wait_all()
        for transfer in self._transfers:
            if transfer.job_id in job_ids:
                transfer.cuda_event.synchronize()


class SsdGpuOffloadingHandlers:
    """Creates GPU<->SSD transfer handlers with staging buffer pool."""

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
        assert gpu_caches
        assert ssd_block_size % gpu_block_size == 0

        # Parse GPU tensor layout (same probe pattern as CpuGpuOffloadingHandlers)
        kernel_block_size: int | None = None
        parsed_gpu_tensors: list[tuple[torch.Tensor, bool]] = []

        for layer_name, gpu_tensor in gpu_caches.items():
            gpu_shape = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]
            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256
            )

            split_k_and_v = False
            has_layers_dim = False
            if len(gpu_shape) != len(test_shape):
                assert len(gpu_shape) == len(test_shape) + 1
                has_layers_dim = True
                test_shape = (80,) + test_shape
            elif test_shape[0] != 1234:
                assert test_shape[0] == 2
                assert test_shape[1] == 1234
                assert gpu_shape[0] == 2
                split_k_and_v = True

            try:
                kv_cache_stride_order = (
                    attn_backend.get_kv_cache_stride_order(
                        include_num_layers_dimension=has_layers_dim
                    )
                )
                assert len(kv_cache_stride_order) == len(gpu_shape)
            except (AttributeError, NotImplementedError):
                kv_cache_stride_order = tuple(range(len(gpu_shape)))

            test_shape = tuple(
                test_shape[i] for i in kv_cache_stride_order
            )

            block_size_idx = test_shape.index(16)
            if kernel_block_size is not None:
                assert kernel_block_size == gpu_shape[block_size_idx]
            else:
                kernel_block_size = gpu_shape[block_size_idx]
                assert gpu_block_size % kernel_block_size == 0

            parsed_gpu_tensors.append((gpu_tensor, split_k_and_v))

        assert kernel_block_size is not None
        gpu_block_size_factor = gpu_block_size // kernel_block_size

        # Flatten GPU tensors (unbind split K/V)
        gpu_tensors: list[torch.Tensor] = []
        for gpu_tensor, split_k_and_v in parsed_gpu_tensors:
            gpu_tensors.extend(
                gpu_tensor.unbind(0) if split_k_and_v else [gpu_tensor]
            )

        # Calculate per-sub-block sizes
        block_size_in_bytes = [
            tensor.element_size() * tensor.stride(0)
            for tensor in gpu_tensors
        ]

        # Initialize SSD store (C++ backend)
        ssd_store = None
        try:
            import vllm._ssd_kv_C as ssd_kv

            # Create file paths for each ssd_path
            file_paths = []
            for i, path in enumerate(ssd_paths):
                file_paths.append(f"{path}_{i}.dat")

            ssd_store_handle = ssd_kv.create(
                file_paths=file_paths,
                num_blocks=num_ssd_blocks,
                block_bytes=block_bytes,
                page_size=page_size,
                io_queue_depth=io_queue_depth,
            )

            # Wrap the C module in a simple object for the handlers
            class SSDStoreWrapper:
                def __init__(self, handle, module):
                    self._handle = handle
                    self._mod = module

                def write_block(self, block_id, buffer_ptr, job_id):
                    # For now, create a tensor view at the buffer pointer
                    self._mod.write_blocks(
                        self._handle,
                        torch.tensor([block_id], dtype=torch.int64),
                        torch.tensor([0], dtype=torch.uint8),  # placeholder
                        job_id,
                    )

                def read_block(self, block_id, buffer_ptr, job_id):
                    self._mod.read_blocks(
                        self._handle,
                        torch.tensor([block_id], dtype=torch.int64),
                        torch.tensor([0], dtype=torch.uint8),  # placeholder
                        job_id,
                    )

                def poll(self):
                    return self._mod.poll(self._handle)

                def wait_all(self):
                    return self._mod.wait_all(self._handle)

            ssd_store = SSDStoreWrapper(ssd_store_handle, ssd_kv)
            logger.info(
                "SSD KV store initialized: %d blocks, %d bytes/block, "
                "%d files",
                num_ssd_blocks,
                block_bytes,
                len(file_paths),
            )
        except ImportError:
            logger.warning(
                "vllm._ssd_kv_C not available. SSD offloading will use "
                "fallback (synchronous file I/O). Install liburing-dev and "
                "rebuild vLLM for io_uring support."
            )
            ssd_store = FallbackSSDStore(
                ssd_paths, num_ssd_blocks, block_bytes
            )

        # Calculate staging buffer size per slot
        # Each slot holds all KV data for block_size_factor sub-blocks
        ssd_block_size_factor = ssd_block_size // kernel_block_size
        staging_bytes_per_slot = sum(block_size_in_bytes) * ssd_block_size_factor

        # Align to page size for O_DIRECT
        if staging_bytes_per_slot % page_size != 0:
            staging_bytes_per_slot = (
                (staging_bytes_per_slot + page_size - 1)
                // page_size
                * page_size
            )

        staging_pool = StagingBufferPool(
            num_slots=max_concurrent_transfers,
            bytes_per_slot=staging_bytes_per_slot,
            gpu_tensors=gpu_tensors,
            block_size_in_bytes=block_size_in_bytes,
        )

        logger.info(
            "SSD staging buffer pool: %d slots, %d bytes/slot (%.1f MB total)",
            max_concurrent_transfers,
            staging_bytes_per_slot,
            max_concurrent_transfers * staging_bytes_per_slot / (1024**2),
        )

        self.gpu_to_ssd_handler = GpuToSsdHandler(
            gpu_tensors=gpu_tensors,
            gpu_block_size_factor=gpu_block_size_factor,
            staging_pool=staging_pool,
            ssd_store=ssd_store,
            block_size_in_bytes=block_size_in_bytes,
        )

        self.ssd_to_gpu_handler = SsdToGpuHandler(
            gpu_tensors=gpu_tensors,
            gpu_block_size_factor=gpu_block_size_factor,
            staging_pool=staging_pool,
            ssd_store=ssd_store,
            block_size_in_bytes=block_size_in_bytes,
        )


class FallbackSSDStore:
    """Synchronous file I/O fallback when io_uring is not available."""

    def __init__(
        self,
        ssd_paths: list[str],
        num_blocks: int,
        block_bytes: int,
    ):
        import os

        self._block_bytes = block_bytes
        self._num_files = len(ssd_paths)
        self._fds: list[int] = []
        self._results: list[tuple[int, bool, bool]] = []

        blocks_per_file = (num_blocks + self._num_files - 1) // self._num_files
        file_size = blocks_per_file * block_bytes

        for i, path in enumerate(ssd_paths):
            filepath = f"{path}_{i}.dat"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fd = os.open(filepath, os.O_RDWR | os.O_CREAT, 0o644)
            os.ftruncate(fd, file_size)
            self._fds.append(fd)

    def _block_location(self, block_id: int) -> tuple[int, int]:
        file_idx = block_id % self._num_files
        offset = (block_id // self._num_files) * self._block_bytes
        return file_idx, offset

    def write_block(self, block_id: int, buffer_ptr: int, job_id: int):
        import ctypes
        import os

        file_idx, offset = self._block_location(block_id)
        data = (ctypes.c_char * self._block_bytes).from_address(buffer_ptr)
        os.pwrite(self._fds[file_idx], data, offset)
        self._results.append((job_id, True, False))

    def read_block(self, block_id: int, buffer_ptr: int, job_id: int):
        import ctypes
        import os

        file_idx, offset = self._block_location(block_id)
        data = os.pread(self._fds[file_idx], self._block_bytes, offset)
        ctypes.memmove(buffer_ptr, data, len(data))
        self._results.append((job_id, True, True))

    def poll(self) -> list[tuple[int, bool, bool]]:
        results = self._results
        self._results = []
        return results

    def wait_all(self) -> list[tuple[int, bool, bool]]:
        return self.poll()
