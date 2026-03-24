// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "io_uring_engine.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

namespace vllm::ssd_kv {

IOUringEngine::IOUringEngine(
    size_t page_size,
    size_t max_read_queue_depth,
    size_t max_write_queue_depth)
    : page_size_(page_size),
      max_read_queue_depth_(max_read_queue_depth),
      max_write_queue_depth_(max_write_queue_depth) {
  // Initialize read ring
  int ret = io_uring_queue_init(max_read_queue_depth_, &read_ring_, 0);
  if (ret != 0) {
    throw std::runtime_error(
        "Failed to init io_uring read ring: " + std::string(strerror(-ret)));
  }
  read_ring_initialized_ = true;

  // Initialize write ring
  ret = io_uring_queue_init(max_write_queue_depth_, &write_ring_, 0);
  if (ret != 0) {
    io_uring_queue_exit(&read_ring_);
    read_ring_initialized_ = false;
    throw std::runtime_error(
        "Failed to init io_uring write ring: " + std::string(strerror(-ret)));
  }
  write_ring_initialized_ = true;

  // Pre-allocate page-aligned read buffers
  read_buffers_.reserve(max_read_queue_depth_);
  for (size_t i = 0; i < max_read_queue_depth_; i++) {
    read_buffers_.emplace_back(page_size_);
  }
}

IOUringEngine::~IOUringEngine() {
  if (write_ring_initialized_) {
    io_uring_queue_exit(&write_ring_);
  }
  if (read_ring_initialized_) {
    io_uring_queue_exit(&read_ring_);
  }
}

void IOUringEngine::enqueue_read(
    int fd, uint64_t offset, uint32_t length, void* buffer,
    int64_t user_data) {
  if (pending_read_count_ >= max_read_queue_depth_) {
    throw std::runtime_error(
        "io_uring read queue is full, call submit_reads first");
  }

  struct io_uring_sqe* sqe = io_uring_get_sqe(&read_ring_);
  if (!sqe) {
    throw std::runtime_error("Failed to get io_uring SQE for read");
  }

  io_uring_prep_read(sqe, fd, buffer, length, offset);
  io_uring_sqe_set_data64(sqe, static_cast<uint64_t>(user_data));
  pending_read_count_++;
}

void IOUringEngine::enqueue_write(
    int fd, uint64_t offset, uint32_t length, const void* buffer,
    int64_t user_data) {
  if (pending_write_count_ >= max_write_queue_depth_) {
    throw std::runtime_error(
        "io_uring write queue is full, call submit_writes first");
  }

  struct io_uring_sqe* sqe = io_uring_get_sqe(&write_ring_);
  if (!sqe) {
    throw std::runtime_error("Failed to get io_uring SQE for write");
  }

  io_uring_prep_write(sqe, fd, buffer, length, offset);
  io_uring_sqe_set_data64(sqe, static_cast<uint64_t>(user_data));
  pending_write_count_++;
}

int IOUringEngine::submit_reads() {
  if (pending_read_count_ == 0) {
    return 0;
  }
  int ret = io_uring_submit(&read_ring_);
  if (ret < 0) {
    throw std::runtime_error(
        "io_uring_submit (read) failed: " + std::string(strerror(-ret)));
  }
  return ret;
}

int IOUringEngine::submit_writes() {
  if (pending_write_count_ == 0) {
    return 0;
  }
  int ret = io_uring_submit(&write_ring_);
  if (ret < 0) {
    throw std::runtime_error(
        "io_uring_submit (write) failed: " + std::string(strerror(-ret)));
  }
  return ret;
}

std::vector<IOCompletion> IOUringEngine::drain_completions(
    struct io_uring& ring, int count) {
  std::vector<IOCompletion> completions;
  completions.reserve(count);

  for (int i = 0; i < count; i++) {
    struct io_uring_cqe* cqe = nullptr;
    int ret = io_uring_wait_cqe(&ring, &cqe);
    if (ret != 0) {
      completions.push_back(
          {-1, ret, false});
      if (cqe) {
        io_uring_cqe_seen(&ring, cqe);
      }
      continue;
    }

    IOCompletion completion;
    completion.user_data = static_cast<int64_t>(io_uring_cqe_get_data64(cqe));
    completion.result = cqe->res;
    completion.success = (cqe->res >= 0);
    completions.push_back(completion);
    io_uring_cqe_seen(&ring, cqe);
  }

  return completions;
}

std::vector<IOCompletion> IOUringEngine::poll_completions(
    struct io_uring& ring) {
  std::vector<IOCompletion> completions;

  while (true) {
    struct io_uring_cqe* cqe = nullptr;
    int ret = io_uring_peek_cqe(&ring, &cqe);
    if (ret != 0 || !cqe) {
      break;  // No more completions available
    }

    IOCompletion completion;
    completion.user_data = static_cast<int64_t>(io_uring_cqe_get_data64(cqe));
    completion.result = cqe->res;
    completion.success = (cqe->res >= 0);
    completions.push_back(completion);
    io_uring_cqe_seen(&ring, cqe);
  }

  return completions;
}

std::vector<IOCompletion> IOUringEngine::submit_reads_and_wait() {
  if (pending_read_count_ == 0) {
    return {};
  }

  int count = static_cast<int>(pending_read_count_);
  int submitted = io_uring_submit_and_wait(&read_ring_, count);
  if (submitted < 0) {
    // Clear the ring on error
    auto completions = drain_completions(read_ring_, count);
    pending_read_count_ = 0;
    return completions;
  }

  auto completions = drain_completions(read_ring_, count);
  pending_read_count_ = 0;
  return completions;
}

std::vector<IOCompletion> IOUringEngine::submit_writes_and_wait() {
  if (pending_write_count_ == 0) {
    return {};
  }

  int count = static_cast<int>(pending_write_count_);
  int submitted = io_uring_submit_and_wait(&write_ring_, count);
  if (submitted < 0) {
    auto completions = drain_completions(write_ring_, count);
    pending_write_count_ = 0;
    return completions;
  }

  auto completions = drain_completions(write_ring_, count);
  pending_write_count_ = 0;
  return completions;
}

std::vector<IOCompletion> IOUringEngine::poll_read_completions() {
  auto completions = poll_completions(read_ring_);
  pending_read_count_ -= completions.size();
  return completions;
}

std::vector<IOCompletion> IOUringEngine::poll_write_completions() {
  auto completions = poll_completions(write_ring_);
  pending_write_count_ -= completions.size();
  return completions;
}

std::vector<IOCompletion> IOUringEngine::wait_read_completions(int count) {
  auto completions = drain_completions(read_ring_, count);
  pending_read_count_ -= completions.size();
  return completions;
}

std::vector<IOCompletion> IOUringEngine::wait_write_completions(int count) {
  auto completions = drain_completions(write_ring_, count);
  pending_write_count_ -= completions.size();
  return completions;
}

}  // namespace vllm::ssd_kv
