// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "ssd_kv_store.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace vllm::ssd_kv {

SSDKVStore::SSDKVStore(
    const std::vector<std::string>& file_paths,
    int64_t num_blocks,
    int64_t block_bytes,
    size_t page_size,
    size_t io_queue_depth)
    : num_blocks_(num_blocks),
      block_bytes_(block_bytes),
      page_size_(page_size),
      num_files_(static_cast<int>(file_paths.size())) {
  if (file_paths.empty()) {
    throw std::runtime_error("SSDKVStore requires at least one file path");
  }
  if (block_bytes % page_size != 0) {
    throw std::runtime_error(
        "block_bytes (" + std::to_string(block_bytes) +
        ") must be a multiple of page_size (" + std::to_string(page_size) +
        ")");
  }

  // Calculate blocks per file and open files
  int64_t blocks_per_file =
      (num_blocks + num_files_ - 1) / num_files_;
  int64_t file_size = blocks_per_file * block_bytes;

  for (const auto& path : file_paths) {
    // Ensure parent directory exists
    auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }

    int fd = open(path.c_str(), O_RDWR | O_CREAT | O_DIRECT, 0644);
    if (fd < 0) {
      // Close already opened fds
      for (int opened_fd : file_fds_) {
        close(opened_fd);
      }
      throw std::runtime_error(
          "Failed to open SSD file " + path + ": " + strerror(errno));
    }

    // Pre-allocate file space
    if (fallocate(fd, 0, 0, file_size) != 0) {
      // fallocate may not be supported on all filesystems, fall back to ftruncate
      if (ftruncate(fd, file_size) != 0) {
        close(fd);
        for (int opened_fd : file_fds_) {
          close(opened_fd);
        }
        throw std::runtime_error(
            "Failed to allocate space for " + path + ": " + strerror(errno));
      }
    }

    file_fds_.push_back(fd);
  }

  // Create the io_uring engine
  io_engine_ = std::make_unique<IOUringEngine>(
      page_size, io_queue_depth, /*max_write_queue_depth=*/64);

  // Start background IO thread
  io_thread_ = std::thread(&SSDKVStore::io_thread_func, this);
}

SSDKVStore::~SSDKVStore() {
  shutdown();
  for (int fd : file_fds_) {
    close(fd);
  }
}

void SSDKVStore::write_block(
    int64_t block_id, const void* src_buffer, int64_t job_id) {
  if (block_id < 0 || block_id >= num_blocks_) {
    throw std::runtime_error(
        "Invalid block_id: " + std::to_string(block_id));
  }

  IORequest req;
  req.type = IORequestType::WRITE;
  req.block_id = block_id;
  req.buffer = const_cast<void*>(src_buffer);
  req.buffer_size = static_cast<uint32_t>(block_bytes_);
  req.job_id = job_id;

  pending_count_.fetch_add(1);
  {
    std::lock_guard<std::mutex> lock(request_mutex_);
    request_queue_.push(req);
  }
  request_cv_.notify_one();
}

void SSDKVStore::read_block(
    int64_t block_id, void* dst_buffer, int64_t job_id) {
  if (block_id < 0 || block_id >= num_blocks_) {
    throw std::runtime_error(
        "Invalid block_id: " + std::to_string(block_id));
  }

  IORequest req;
  req.type = IORequestType::READ;
  req.block_id = block_id;
  req.buffer = dst_buffer;
  req.buffer_size = static_cast<uint32_t>(block_bytes_);
  req.job_id = job_id;

  pending_count_.fetch_add(1);
  {
    std::lock_guard<std::mutex> lock(request_mutex_);
    request_queue_.push(req);
  }
  request_cv_.notify_one();
}

std::vector<IOResult> SSDKVStore::poll() {
  std::lock_guard<std::mutex> lock(result_mutex_);
  std::vector<IOResult> results;
  results.swap(result_queue_);
  return results;
}

std::vector<IOResult> SSDKVStore::wait_all() {
  // Wait until all pending operations complete
  std::unique_lock<std::mutex> lock(wait_mutex_);
  wait_cv_.wait(lock, [this] { return pending_count_.load() == 0; });

  return poll();
}

void SSDKVStore::shutdown() {
  if (shutdown_requested_.exchange(true)) {
    return;  // Already shutting down
  }

  // Send shutdown request
  {
    std::lock_guard<std::mutex> lock(request_mutex_);
    IORequest req;
    req.type = IORequestType::SHUTDOWN;
    req.block_id = 0;
    req.buffer = nullptr;
    req.buffer_size = 0;
    req.job_id = -1;
    request_queue_.push(req);
  }
  request_cv_.notify_one();

  if (io_thread_.joinable()) {
    io_thread_.join();
  }
}

void SSDKVStore::io_thread_func() {
  std::vector<IORequest> batch;

  while (true) {
    // Wait for requests
    {
      std::unique_lock<std::mutex> lock(request_mutex_);
      request_cv_.wait(lock, [this] { return !request_queue_.empty(); });

      // Drain the queue into a local batch
      while (!request_queue_.empty()) {
        batch.push_back(request_queue_.front());
        request_queue_.pop();
      }
    }

    // Check for shutdown
    for (const auto& req : batch) {
      if (req.type == IORequestType::SHUTDOWN) {
        // Process any remaining non-shutdown requests first
        process_batch(batch);
        return;
      }
    }

    process_batch(batch);
    batch.clear();
  }
}

void SSDKVStore::process_batch(std::vector<IORequest>& batch) {
  // Separate reads and writes, skip shutdown requests
  std::vector<IORequest> reads;
  std::vector<IORequest> writes;

  for (auto& req : batch) {
    if (req.type == IORequestType::READ) {
      reads.push_back(req);
    } else if (req.type == IORequestType::WRITE) {
      writes.push_back(req);
    }
  }

  // Process writes
  if (!writes.empty()) {
    for (auto& req : writes) {
      auto [file_idx, offset] = block_location(req.block_id);
      io_engine_->enqueue_write(
          file_fds_[file_idx], offset, req.buffer_size,
          req.buffer, req.job_id);

      if (io_engine_->write_queue_full()) {
        auto completions = io_engine_->submit_writes_and_wait();
        std::lock_guard<std::mutex> lock(result_mutex_);
        for (const auto& c : completions) {
          result_queue_.push_back(
              {c.user_data, c.success, IORequestType::WRITE});
          pending_count_.fetch_sub(1);
        }
        wait_cv_.notify_all();
      }
    }

    // Submit remaining writes
    auto completions = io_engine_->submit_writes_and_wait();
    {
      std::lock_guard<std::mutex> lock(result_mutex_);
      for (const auto& c : completions) {
        result_queue_.push_back(
            {c.user_data, c.success, IORequestType::WRITE});
        pending_count_.fetch_sub(1);
      }
    }
    wait_cv_.notify_all();
  }

  // Process reads
  if (!reads.empty()) {
    for (auto& req : reads) {
      auto [file_idx, offset] = block_location(req.block_id);
      io_engine_->enqueue_read(
          file_fds_[file_idx], offset, req.buffer_size,
          req.buffer, req.job_id);

      if (io_engine_->read_queue_full()) {
        auto completions = io_engine_->submit_reads_and_wait();
        std::lock_guard<std::mutex> lock(result_mutex_);
        for (const auto& c : completions) {
          result_queue_.push_back(
              {c.user_data, c.success, IORequestType::READ});
          pending_count_.fetch_sub(1);
        }
        wait_cv_.notify_all();
      }
    }

    // Submit remaining reads
    auto completions = io_engine_->submit_reads_and_wait();
    {
      std::lock_guard<std::mutex> lock(result_mutex_);
      for (const auto& c : completions) {
        result_queue_.push_back(
            {c.user_data, c.success, IORequestType::READ});
        pending_count_.fetch_sub(1);
      }
    }
    wait_cv_.notify_all();
  }
}

}  // namespace vllm::ssd_kv
