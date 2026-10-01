/* Copyright 2025-2026 The xLLM Authors.

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

#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include "core/common/macros.h"
#include "worker.pb.h"

namespace xllm {

class CollectiveService final : public proto::Collective {
 public:
  explicit CollectiveService(int32_t total_num);
  ~CollectiveService() override = default;

  void Sync(::google::protobuf::RpcController* controller,
            const proto::AddressInfo* request,
            proto::CommUniqueIdList* response,
            ::google::protobuf::Closure* done) override;

  // wait all worker connected
  std::unordered_map<int32_t, std::string> wait();

 private:
  DISALLOW_COPY_AND_ASSIGN(CollectiveService);

  int32_t total_num_ = 0;
  std::mutex mutex_;
  std::unordered_map<int32_t, std::string> addrs_map_;
};

}  // namespace xllm
