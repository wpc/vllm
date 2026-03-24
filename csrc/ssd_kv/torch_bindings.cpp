// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// SSD KV cache store bindings using TORCH_LIBRARY for Stable ABI compat.
// All operations use int64 handles and raw pointers (via int64) to avoid
// torch::Tensor in C++ (not available under Py_LIMITED_API).

#include <Python.h>
#include <torch/library.h>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/registration.h"
#include "ssd_kv_store.h"

namespace vllm::ssd_kv {

// Global store registry
static std::unordered_map<int64_t, std::unique_ptr<SSDKVStore>> g_stores;
static int64_t g_next_handle = 0;

int64_t ssd_kv_store_create(
    const std::vector<std::string>& file_paths,
    int64_t num_blocks,
    int64_t block_bytes,
    int64_t page_size,
    int64_t io_queue_depth) {
  auto store = std::make_unique<SSDKVStore>(
      file_paths, num_blocks, block_bytes,
      static_cast<size_t>(page_size),
      static_cast<size_t>(io_queue_depth));

  int64_t handle = g_next_handle++;
  g_stores[handle] = std::move(store);
  return handle;
}

void ssd_kv_store_destroy(int64_t handle) {
  auto it = g_stores.find(handle);
  if (it != g_stores.end()) {
    g_stores.erase(it);
  }
}

// Write a single block to SSD.
// buffer_ptr: raw pointer to page-aligned source data (as int64)
// block_bytes: number of bytes to write
void ssd_kv_store_write_block(
    int64_t handle,
    int64_t block_id,
    int64_t buffer_ptr,
    int64_t job_id) {
  auto it = g_stores.find(handle);
  if (it == g_stores.end()) {
    throw std::runtime_error("Invalid SSD KV store handle");
  }
  it->second->write_block(
      block_id,
      reinterpret_cast<const void*>(buffer_ptr),
      job_id);
}

// Read a single block from SSD.
// buffer_ptr: raw pointer to page-aligned destination buffer (as int64)
void ssd_kv_store_read_block(
    int64_t handle,
    int64_t block_id,
    int64_t buffer_ptr,
    int64_t job_id) {
  auto it = g_stores.find(handle);
  if (it == g_stores.end()) {
    throw std::runtime_error("Invalid SSD KV store handle");
  }
  it->second->read_block(
      block_id,
      reinterpret_cast<void*>(buffer_ptr),
      job_id);
}

// Poll for completed IO operations.
// Returns results via output arrays (pre-allocated by caller).
// Returns the number of completed operations.
int64_t ssd_kv_store_poll(
    int64_t handle,
    int64_t out_job_ids_ptr,
    int64_t out_success_ptr,
    int64_t out_is_read_ptr,
    int64_t max_results) {
  auto it = g_stores.find(handle);
  if (it == g_stores.end()) {
    throw std::runtime_error("Invalid SSD KV store handle");
  }

  auto results = it->second->poll();
  int64_t count = std::min(static_cast<int64_t>(results.size()), max_results);

  auto* job_ids = reinterpret_cast<int64_t*>(out_job_ids_ptr);
  auto* success = reinterpret_cast<int64_t*>(out_success_ptr);
  auto* is_read = reinterpret_cast<int64_t*>(out_is_read_ptr);

  for (int64_t i = 0; i < count; i++) {
    job_ids[i] = results[i].job_id;
    success[i] = results[i].success ? 1 : 0;
    is_read[i] = (results[i].type == IORequestType::READ) ? 1 : 0;
  }

  return count;
}

// Wait for all pending IO and return count.
int64_t ssd_kv_store_wait_all(int64_t handle) {
  auto it = g_stores.find(handle);
  if (it == g_stores.end()) {
    throw std::runtime_error("Invalid SSD KV store handle");
  }
  auto results = it->second->wait_all();
  return static_cast<int64_t>(results.size());
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("create(str[] file_paths, int num_blocks, int block_bytes, "
          "int page_size, int io_queue_depth) -> int");
  ops.impl("create", &ssd_kv_store_create);

  ops.def("destroy(int handle) -> ()");
  ops.impl("destroy", &ssd_kv_store_destroy);

  ops.def("write_block(int handle, int block_id, int buffer_ptr, "
          "int job_id) -> ()");
  ops.impl("write_block", &ssd_kv_store_write_block);

  ops.def("read_block(int handle, int block_id, int buffer_ptr, "
          "int job_id) -> ()");
  ops.impl("read_block", &ssd_kv_store_read_block);

  ops.def("poll(int handle, int out_job_ids_ptr, int out_success_ptr, "
          "int out_is_read_ptr, int max_results) -> int");
  ops.impl("poll", &ssd_kv_store_poll);

  ops.def("wait_all(int handle) -> int");
  ops.impl("wait_all", &ssd_kv_store_wait_all);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

}  // namespace vllm::ssd_kv
