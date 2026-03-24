// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/library.h>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/registration.h"
#include "ssd_kv_store.h"

namespace vllm::ssd_kv {

// Global store registry (handles are just integer IDs)
static std::unordered_map<int64_t, std::unique_ptr<SSDKVStore>> g_stores;
static int64_t g_next_handle = 0;

// Create an SSD KV store. Returns a handle (int64).
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

// Write blocks from a contiguous pinned CPU tensor to SSD.
void ssd_kv_store_write_blocks(
    int64_t handle,
    torch::Tensor block_ids,
    torch::Tensor src_buffer,
    int64_t job_id_start) {
  auto it = g_stores.find(handle);
  TORCH_CHECK(it != g_stores.end(), "Invalid SSD KV store handle");
  auto& store = it->second;

  TORCH_CHECK(block_ids.dim() == 1 && block_ids.dtype() == torch::kInt64,
              "block_ids must be 1D int64 tensor");
  TORCH_CHECK(src_buffer.is_contiguous(), "src_buffer must be contiguous");
  TORCH_CHECK(!src_buffer.is_cuda(), "src_buffer must be on CPU");

  int64_t num_blocks = block_ids.size(0);
  int64_t block_bytes = store->block_bytes();
  TORCH_CHECK(
      src_buffer.nbytes() >= static_cast<size_t>(num_blocks * block_bytes),
      "src_buffer too small");

  auto* block_ids_ptr = block_ids.data_ptr<int64_t>();
  auto* src_ptr = static_cast<uint8_t*>(src_buffer.data_ptr());

  for (int64_t i = 0; i < num_blocks; i++) {
    store->write_block(
        block_ids_ptr[i],
        src_ptr + i * block_bytes,
        job_id_start + i);
  }
}

// Read blocks from SSD into a contiguous pinned CPU tensor.
void ssd_kv_store_read_blocks(
    int64_t handle,
    torch::Tensor block_ids,
    torch::Tensor dst_buffer,
    int64_t job_id_start) {
  auto it = g_stores.find(handle);
  TORCH_CHECK(it != g_stores.end(), "Invalid SSD KV store handle");
  auto& store = it->second;

  TORCH_CHECK(block_ids.dim() == 1 && block_ids.dtype() == torch::kInt64,
              "block_ids must be 1D int64 tensor");
  TORCH_CHECK(dst_buffer.is_contiguous(), "dst_buffer must be contiguous");
  TORCH_CHECK(!dst_buffer.is_cuda(), "dst_buffer must be on CPU");

  int64_t num_blocks = block_ids.size(0);
  int64_t block_bytes = store->block_bytes();
  TORCH_CHECK(
      dst_buffer.nbytes() >= static_cast<size_t>(num_blocks * block_bytes),
      "dst_buffer too small");

  auto* block_ids_ptr = block_ids.data_ptr<int64_t>();
  auto* dst_ptr = static_cast<uint8_t*>(dst_buffer.data_ptr());

  for (int64_t i = 0; i < num_blocks; i++) {
    store->read_block(
        block_ids_ptr[i],
        dst_ptr + i * block_bytes,
        job_id_start + i);
  }
}

// Poll for completed IO operations.
// Returns a tensor of shape (N, 3) with columns [job_id, success, is_read].
torch::Tensor ssd_kv_store_poll(int64_t handle) {
  auto it = g_stores.find(handle);
  TORCH_CHECK(it != g_stores.end(), "Invalid SSD KV store handle");

  auto results = it->second->poll();
  auto out = torch::empty({static_cast<int64_t>(results.size()), 3},
                          torch::kInt64);
  auto* out_ptr = out.data_ptr<int64_t>();
  for (size_t i = 0; i < results.size(); i++) {
    out_ptr[i * 3] = results[i].job_id;
    out_ptr[i * 3 + 1] = results[i].success ? 1 : 0;
    out_ptr[i * 3 + 2] = (results[i].type == IORequestType::READ) ? 1 : 0;
  }
  return out;
}

// Wait for all pending IO operations and return results.
torch::Tensor ssd_kv_store_wait_all(int64_t handle) {
  auto it = g_stores.find(handle);
  TORCH_CHECK(it != g_stores.end(), "Invalid SSD KV store handle");

  auto results = it->second->wait_all();
  auto out = torch::empty({static_cast<int64_t>(results.size()), 3},
                          torch::kInt64);
  auto* out_ptr = out.data_ptr<int64_t>();
  for (size_t i = 0; i < results.size(); i++) {
    out_ptr[i * 3] = results[i].job_id;
    out_ptr[i * 3 + 1] = results[i].success ? 1 : 0;
    out_ptr[i * 3 + 2] = (results[i].type == IORequestType::READ) ? 1 : 0;
  }
  return out;
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("create(str[] file_paths, int num_blocks, int block_bytes, "
          "int page_size=4096, int io_queue_depth=256) -> int");
  ops.impl("create", torch::kCPU, &ssd_kv_store_create);

  ops.def("destroy(int handle) -> ()");
  ops.impl("destroy", torch::kCPU, &ssd_kv_store_destroy);

  ops.def("write_blocks(int handle, Tensor block_ids, Tensor src_buffer, "
          "int job_id_start) -> ()");
  ops.impl("write_blocks", torch::kCPU, &ssd_kv_store_write_blocks);

  ops.def("read_blocks(int handle, Tensor block_ids, Tensor dst_buffer, "
          "int job_id_start) -> ()");
  ops.impl("read_blocks", torch::kCPU, &ssd_kv_store_read_blocks);

  ops.def("poll(int handle) -> Tensor");
  ops.impl("poll", torch::kCPU, &ssd_kv_store_poll);

  ops.def("wait_all(int handle) -> Tensor");
  ops.impl("wait_all", torch::kCPU, &ssd_kv_store_wait_all);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

}  // namespace vllm::ssd_kv
