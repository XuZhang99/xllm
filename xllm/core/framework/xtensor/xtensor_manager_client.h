/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <folly/futures/Future.h>

#include <memory>

#include "util/threadpool.h"
#include "xtensor_manager.h"

namespace xllm {
class XTensorManagerClient {
 public:
  XTensorManagerClient() = default;
  explicit XTensorManagerClient(XTensorManager* p) : xtensor_manager_(p) {}
  virtual ~XTensorManagerClient() = default;

  bool allocate(int32_t& seq_id, size_t num_tokens);
  void deallocate(int32_t seq_id);

  folly::SemiFuture<bool> allocate_async(int32_t& seq_id, size_t num_tokens);
  folly::SemiFuture<folly::Unit> deallocate_async(int32_t seq_id);

  size_t num_free_pages_per_layer() const;
  size_t num_used_pages_per_layer() const;
  double kv_cache_utilization() const;

  folly::SemiFuture<size_t> num_free_pages_per_layer_async();
  folly::SemiFuture<size_t> num_used_pages_per_layer_async();

 private:
  XTensorManager* xtensor_manager_ = nullptr;
};
}  // namespace xllm