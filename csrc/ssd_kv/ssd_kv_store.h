// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "io_uring_engine.h"

namespace vllm::ssd_kv {

// Request types for the background IO thread
enum class IORequestType { READ, WRITE, SHUTDOWN };

struct IORequest {
  IORequestType type;
  int64_t block_id;      // SSD block slot index
  void* buffer;          // source (write) or destination (read) buffer
  uint32_t buffer_size;  // bytes to transfer
  int64_t job_id;        // user-provided job ID for tracking
};

struct IOResult {
  int64_t job_id;
  bool success;
  IORequestType type;
};

// Block-level KV cache store on SSD.
//
// Manages one or more pre-allocated files on SSD mount points.
// Each file is opened with O_DIRECT for DMA bypass.
// Blocks are stored at fixed offsets: block i is at
//   file[i % num_files], offset = (i / num_files) * block_bytes.
//
// All I/O is performed by a background thread using IOUringEngine.
class SSDKVStore {
 public:
  // Create an SSD KV store.
  // file_paths: list of SSD file paths (will be created/truncated)
  // num_blocks: total number of block slots
  // block_bytes: bytes per block (must be page-aligned)
  // page_size: filesystem page size for O_DIRECT alignment (default 4096)
  // io_queue_depth: io_uring queue depth for reads (default 256)
  SSDKVStore(
      const std::vector<std::string>& file_paths,
      int64_t num_blocks,
      int64_t block_bytes,
      size_t page_size = 4096,
      size_t io_queue_depth = 256);

  ~SSDKVStore();

  // Non-copyable, non-movable
  SSDKVStore(const SSDKVStore&) = delete;
  SSDKVStore& operator=(const SSDKVStore&) = delete;

  // Async write a block to SSD.
  // src_buffer must be page-aligned and valid until the write completes.
  void write_block(int64_t block_id, const void* src_buffer, int64_t job_id);

  // Async read a block from SSD.
  // dst_buffer must be page-aligned and valid until the read completes.
  void read_block(int64_t block_id, void* dst_buffer, int64_t job_id);

  // Poll for completed IO operations (non-blocking).
  std::vector<IOResult> poll();

  // Wait for all pending IO operations to complete.
  std::vector<IOResult> wait_all();

  // Shutdown the background IO thread.
  void shutdown();

  int64_t num_blocks() const { return num_blocks_; }
  int64_t block_bytes() const { return block_bytes_; }

 private:
  void io_thread_func();
  void process_batch(std::vector<IORequest>& batch);

  // Map block_id to (file_index, file_offset)
  std::pair<int, uint64_t> block_location(int64_t block_id) const {
    int file_idx = static_cast<int>(block_id % num_files_);
    uint64_t offset =
        static_cast<uint64_t>(block_id / num_files_) * block_bytes_;
    return {file_idx, offset};
  }

  const int64_t num_blocks_;
  const int64_t block_bytes_;
  const size_t page_size_;
  const int num_files_;

  std::vector<int> file_fds_;  // file descriptors opened with O_DIRECT
  std::unique_ptr<IOUringEngine> io_engine_;

  // Background IO thread
  std::thread io_thread_;
  std::mutex request_mutex_;
  std::condition_variable request_cv_;
  std::queue<IORequest> request_queue_;

  // Results from completed IO operations
  std::mutex result_mutex_;
  std::vector<IOResult> result_queue_;

  std::atomic<bool> shutdown_requested_{false};
  std::atomic<int64_t> pending_count_{0};
  std::mutex wait_mutex_;
  std::condition_variable wait_cv_;
};

}  // namespace vllm::ssd_kv
