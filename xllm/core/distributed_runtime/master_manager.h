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

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/macros.h"
#include "core/common/types.h"

namespace xllm {

class Master;

// Owns and routes the masters hosted by one API service process. A model id is
// the public routing key at present; a later instance id can be added without
// changing the ownership and lifecycle contract.
class MasterManager final {
 public:
  using MasterHandle = std::shared_ptr<Master>;

  MasterManager() = default;
  ~MasterManager();

  // Registers a master and transfers ownership to the manager. Registration
  // fails when model_id is empty, master is null, or the key already exists.
  bool register_master(std::string model_id, std::unique_ptr<Master> master);

  // Registers a non-owning master. This is used for the process' primary
  // master, whose lifetime is still owned by xllm.cpp during the migration.
  bool register_external_master(const std::string& model_id, Master* master);

  bool has_master(const std::string& model_id) const;
  MasterHandle find_master(const std::string& model_id) const;
  MasterHandle default_master() const;
  std::vector<std::string> model_ids() const;

  bool set_default_model(const std::string& model_id);

  // Transitions an LLM master to a sleep state after confirming that it has
  // no in-flight requests.
  bool sleep(const std::string& model_id,
             MasterStatus master_status,
             std::string* error_message);

  // Wakes an LLM master and restores request admission after the engine is
  // ready to serve requests again.
  bool wakeup(const std::string& model_id,
              const WakeupOptions& options,
              std::string* error_message);

  // Releases all registered masters. Manager-owned masters are destroyed after
  // the registry is detached, allowing their destructors to stop worker
  // threads without holding the registry lock. External masters are
  // represented by no-op deleters and remain owned by their original owner.
  void shutdown();

 private:
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, MasterHandle> masters_;
  std::string default_model_;
  mutable std::mutex lifecycle_mutex_;
  bool shutdown_started_ = false;

  DISALLOW_COPY_AND_ASSIGN(MasterManager);
};

}  // namespace xllm
