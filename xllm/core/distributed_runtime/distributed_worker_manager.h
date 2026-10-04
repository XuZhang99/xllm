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

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "core/common/macros.h"
#include "core/runtime/options.h"
#include "core/runtime/worker_client.h"

namespace xllm {

class WorkerServer;

// Owns worker servers, cluster rendezvous, and worker clients. Multiple engines
// can share the manager to use the same distributed workers.
class DistributedWorkerManager final {
 public:
  explicit DistributedWorkerManager(const runtime::Options& options);
  ~DistributedWorkerManager();

  // Only the leader node creates clients for all global ranks.
  const std::vector<std::shared_ptr<WorkerClient>>& get_worker_clients() const {
    return worker_clients_;
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(DistributedWorkerManager);

  void start_worker_servers(const runtime::Options& options,
                            const std::string& master_node_addr);
  void connect_worker_clients(const runtime::Options& options,
                              const std::string& master_node_addr);
  void wait_for_worker_servers() const;
  void start_health_checks();

  std::string collective_server_name_;
  std::vector<std::shared_ptr<WorkerClient>> worker_clients_;
  // Worker threads borrow these flags; keep them alive until servers stop.
  std::vector<std::atomic<bool>> worker_ready_;
  std::vector<std::unique_ptr<WorkerServer>> worker_servers_;
};

}  // namespace xllm
