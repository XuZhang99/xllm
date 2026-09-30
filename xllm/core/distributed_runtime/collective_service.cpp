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

#include "core/distributed_runtime/collective_service.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/closure_guard.h>

namespace xllm {

// Communicators rendezvous through their own stores. Creating unused HCCL
// root infos here leaves topology-server threads waiting during process exit.
CollectiveService::CollectiveService(int32_t total_num)
    : total_num_(total_num) {}

void CollectiveService::Sync(::google::protobuf::RpcController* /*controller*/,
                             const proto::AddressInfo* request,
                             proto::CommUniqueIdList* /*response*/,
                             ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  std::string address = request->address();
  int32_t global_rank = request->global_rank();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    addrs_map_[global_rank] = address;
  }
}

std::unordered_map<int32_t, std::string> CollectiveService::wait() {
  int connected = 0;
  while (connected < total_num_) {
    absl::SleepFor(absl::Milliseconds(1000));
    {
      std::lock_guard<std::mutex> lock(mutex_);
      connected = addrs_map_.size();
    }
  }

  return addrs_map_;
}

}  // namespace xllm
