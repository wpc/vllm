// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <liburing.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace vllm::ssd_kv {

// RAII wrapper for page-aligned memory (required for O_DIRECT)
class AlignedBuffer {
 public:
  explicit AlignedBuffer(size_t size, size_t alignment = 4096)
      : size_(size), alignment_(alignment) {
    if (posix_memalign(&data_, alignment, size) != 0) {
      throw std::runtime_error(
          "posix_memalign failed for size=" + std::to_string(size));
    }
  }

  ~AlignedBuffer() {
    if (data_) {
      free(data_);
    }
  }

  // Non-copyable
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  // Movable
  AlignedBuffer(AlignedBuffer&& other) noexcept
      : data_(other.data_), size_(other.size_), alignment_(other.alignment_) {
    other.data_ = nullptr;
  }

  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
      if (data_) {
        free(data_);
      }
      data_ = other.data_;
      size_ = other.size_;
      alignment_ = other.alignment_;
      other.data_ = nullptr;
    }
    return *this;
  }

  void* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  void* data_{nullptr};
  size_t size_{0};
  size_t alignment_{4096};
};

// Completion result from io_uring operations
struct IOCompletion {
  int64_t user_data;  // user-provided ID for this IO operation
  int32_t result;     // result code (bytes transferred or negative errno)
  bool success;
};

// io_uring engine for async file I/O
// Not thread-safe — use one instance per thread or protect with a mutex.
//
// Adapted from caffe2::embedding_db::IOBackend but uses pure stdlib
// (no folly dependency).
class IOUringEngine {
 public:
  IOUringEngine(
      size_t page_size,
      size_t max_read_queue_depth = 256,
      size_t max_write_queue_depth = 64);

  ~IOUringEngine();

  // Non-copyable, non-movable
  IOUringEngine(const IOUringEngine&) = delete;
  IOUringEngine& operator=(const IOUringEngine&) = delete;
  IOUringEngine(IOUringEngine&&) = delete;
  IOUringEngine& operator=(IOUringEngine&&) = delete;

  // Enqueue a read operation. The buffer must be page-aligned and remain
  // valid until submit_reads() + wait_reads() complete.
  // user_data is returned in the IOCompletion for identification.
  void enqueue_read(int fd, uint64_t offset, uint32_t length, void* buffer,
                    int64_t user_data);

  // Enqueue a write operation. The buffer must be page-aligned and remain
  // valid until submit_writes() + wait_writes() complete.
  void enqueue_write(int fd, uint64_t offset, uint32_t length,
                     const void* buffer, int64_t user_data);

  // Submit all queued reads and wait for all completions.
  // Returns completions for all submitted reads.
  std::vector<IOCompletion> submit_reads_and_wait();

  // Submit all queued writes and wait for all completions.
  std::vector<IOCompletion> submit_writes_and_wait();

  // Submit all queued reads (non-blocking).
  // Returns the number of submitted entries.
  int submit_reads();

  // Submit all queued writes (non-blocking).
  int submit_writes();

  // Poll for completed reads (non-blocking).
  std::vector<IOCompletion> poll_read_completions();

  // Poll for completed writes (non-blocking).
  std::vector<IOCompletion> poll_write_completions();

  // Wait for exactly `count` read completions (blocking).
  std::vector<IOCompletion> wait_read_completions(int count);

  // Wait for exactly `count` write completions (blocking).
  std::vector<IOCompletion> wait_write_completions(int count);

  size_t pending_reads() const { return pending_read_count_; }
  size_t pending_writes() const { return pending_write_count_; }
  bool read_queue_full() const {
    return pending_read_count_ >= max_read_queue_depth_;
  }
  bool write_queue_full() const {
    return pending_write_count_ >= max_write_queue_depth_;
  }

  // Get a pre-allocated read buffer by index.
  // The buffer pool has max_read_queue_depth buffers, each of page_size bytes.
  AlignedBuffer& get_read_buffer(size_t index) { return read_buffers_[index]; }
  size_t page_size() const { return page_size_; }

 private:
  std::vector<IOCompletion> drain_completions(struct io_uring& ring, int count);
  std::vector<IOCompletion> poll_completions(struct io_uring& ring);

  const size_t page_size_;
  const size_t max_read_queue_depth_;
  const size_t max_write_queue_depth_;

  struct io_uring read_ring_{};
  struct io_uring write_ring_{};
  bool read_ring_initialized_{false};
  bool write_ring_initialized_{false};

  size_t pending_read_count_{0};
  size_t pending_write_count_{0};

  // Pre-allocated page-aligned read buffers
  std::vector<AlignedBuffer> read_buffers_;
};

}  // namespace vllm::ssd_kv
