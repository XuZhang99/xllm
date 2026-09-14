/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/platform/npu/mapped_host_memory.h"

#include <acl/acl.h>
#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>
#include <sys/mman.h>
#include <torch_npu/csrc/core/NPUStorageImpl.h>
#include <unistd.h>

#include <limits>
#include <memory>

namespace xllm {
namespace {

class MappedHostMemory final {
 public:
  explicit MappedHostMemory(size_t bytes) : bytes_(bytes) {
    CHECK_EQ(aclrtGetDevice(&device_id_), ACL_SUCCESS);
    host_ = mmap(nullptr,
                 bytes_,
                 PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS,
                 -1,
                 0);
    CHECK(host_ != MAP_FAILED) << "HiSparse Host mmap failed";
    // Registration pins and maps the pages for device access. Do not also
    // mlock them: that imposes a second, unrelated RLIMIT_MEMLOCK budget.
    CHECK_EQ(
        aclrtHostRegister(host_, bytes_, ACL_HOST_REGISTER_MAPPED, &mapped_),
        ACL_SUCCESS)
        << "HiSparse Host mapping is unsupported on this device";
  }

  ~MappedHostMemory() {
    // All execution streams must have completed before cache destruction.
    int32_t current_device = -1;
    CHECK_EQ(aclrtGetDevice(&current_device), ACL_SUCCESS);
    CHECK_EQ(aclrtSetDevice(device_id_), ACL_SUCCESS);
    CHECK_EQ(aclrtHostUnregister(host_), ACL_SUCCESS);
    CHECK_EQ(munmap(host_, bytes_), 0);
    CHECK_EQ(aclrtSetDevice(current_device), ACL_SUCCESS);
  }

  void* data() const { return mapped_; }

 private:
  size_t bytes_;
  int32_t device_id_ = -1;
  void* host_ = nullptr;
  void* mapped_ = nullptr;
};

void delete_mapped_host_memory(void* context) {
  delete static_cast<MappedHostMemory*>(context);
}

}  // namespace

torch::Tensor allocate_mapped_host_tensor(const std::vector<int64_t>& shape,
                                          torch::ScalarType dtype,
                                          const torch::Device& device) {
  CHECK(device.type() == torch::kPrivateUse1);
  const torch::DeviceGuard device_guard(device);
  size_t bytes = torch::elementSize(dtype);
  for (const int64_t dim : shape) {
    CHECK_GT(dim, 0);
    CHECK_LE(static_cast<size_t>(dim),
             std::numeric_limits<size_t>::max() / bytes);
    bytes *= static_cast<size_t>(dim);
  }
  auto tensor = torch::empty({0}, torch::dtype(dtype).device(device));
  auto memory = std::make_unique<MappedHostMemory>(bytes);
  void* mapped = memory->data();
  torch::DataPtr data(
      mapped, memory.release(), delete_mapped_host_memory, device);
  auto* storage_create = torch::GetStorageImplCreate(torch::kPrivateUse1);
  auto* allocator = torch::GetAllocator(torch::kPrivateUse1);
  torch::Storage storage = storage_create(torch::StorageImpl::use_byte_size_t(),
                                          torch::SymInt(bytes),
                                          std::move(data),
                                          allocator,
                                          false);
  tensor.set_(storage, 0, shape);
  auto* npu_storage = static_cast<torch_npu::NPUStorageImpl*>(
      tensor.storage().unsafeGetStorageImpl());
  npu_storage->npu_desc_.npu_format_ = ACL_FORMAT_ND;
  return tensor;
}

}  // namespace xllm
