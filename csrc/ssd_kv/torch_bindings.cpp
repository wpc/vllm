// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ssd_kv_store.h"

namespace py = pybind11;

namespace vllm::ssd_kv {

// Global store registry (handles are just integer IDs)
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

// Write blocks from a contiguous pinned CPU tensor to SSD.
// block_ids: 1D int64 tensor of SSD block slot indices
// src_buffer: contiguous CPU tensor (pinned memory) with all blocks' data
//   concatenated. Size must be len(block_ids) * block_bytes.
// job_id_start: starting job ID; each block gets job_id_start + i.
void ssd_kv_store_write_blocks(
    int64_t handle,
    torch::Tensor block_ids,
    torch::Tensor src_buffer,
    int64_t job_id_start) {
  auto it = g_stores.find(handle);
  if (it == g_stores.end()) {
    throw std::runtime_error("Invalid SSD KV store handle");
  }
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
// block_ids: 1D int64 tensor of SSD block slot indices
// dst_buffer: contiguous CPU tensor (pinned memory) to receive block data.
//   Size must be len(block_ids) * block_bytes.
// job_id_start: starting job ID; each block gets job_id_start + i.
void ssd_kv_store_read_blocks(
    int64_t handle,
    torch::Tensor block_ids,
    torch::Tensor dst_buffer,
    int64_t job_id_start) {
  auto it = g_stores.find(handle);
  if (it == g_stores.end()) {
    throw std::runtime_error("Invalid SSD KV store handle");
  }
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
// Returns list of (job_id, success, is_read) tuples.
py::list ssd_kv_store_poll(int64_t handle) {
  auto it = g_stores.find(handle);
  if (it == g_stores.end()) {
    throw std::runtime_error("Invalid SSD KV store handle");
  }

  auto results = it->second->poll();
  py::list out;
  for (const auto& r : results) {
    out.append(py::make_tuple(
        r.job_id, r.success, r.type == IORequestType::READ));
  }
  return out;
}

// Wait for all pending IO operations and return results.
py::list ssd_kv_store_wait_all(int64_t handle) {
  auto it = g_stores.find(handle);
  if (it == g_stores.end()) {
    throw std::runtime_error("Invalid SSD KV store handle");
  }

  auto results = it->second->wait_all();
  py::list out;
  for (const auto& r : results) {
    out.append(py::make_tuple(
        r.job_id, r.success, r.type == IORequestType::READ));
  }
  return out;
}

PYBIND11_MODULE(_ssd_kv_C, m) {
  m.doc() = "SSD KV cache store with io_uring backend";

  m.def("create", &ssd_kv_store_create,
        "Create an SSD KV store",
        py::arg("file_paths"),
        py::arg("num_blocks"),
        py::arg("block_bytes"),
        py::arg("page_size") = 4096,
        py::arg("io_queue_depth") = 256);

  m.def("destroy", &ssd_kv_store_destroy,
        "Destroy an SSD KV store",
        py::arg("handle"));

  m.def("write_blocks", &ssd_kv_store_write_blocks,
        "Write blocks to SSD",
        py::arg("handle"),
        py::arg("block_ids"),
        py::arg("src_buffer"),
        py::arg("job_id_start"));

  m.def("read_blocks", &ssd_kv_store_read_blocks,
        "Read blocks from SSD",
        py::arg("handle"),
        py::arg("block_ids"),
        py::arg("dst_buffer"),
        py::arg("job_id_start"));

  m.def("poll", &ssd_kv_store_poll,
        "Poll for completed IO operations",
        py::arg("handle"));

  m.def("wait_all", &ssd_kv_store_wait_all,
        "Wait for all pending IO operations",
        py::arg("handle"));
}

}  // namespace vllm::ssd_kv
