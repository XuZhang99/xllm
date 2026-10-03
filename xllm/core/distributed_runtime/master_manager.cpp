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

#include "core/distributed_runtime/master_manager.h"

#include <mutex>
#include <utility>

#include "core/distributed_runtime/master.h"

namespace xllm {

namespace {

MasterManager::MasterHandle make_external_master_handle(Master* master) {
  return MasterManager::MasterHandle(master, [](Master* /*unused*/) {});
}

}  // namespace

MasterManager::~MasterManager() { shutdown(); }

bool MasterManager::register_master(std::string model_id,
                                    std::unique_ptr<Master> master) {
  if (model_id.empty() || master == nullptr) {
    return false;
  }

  MasterHandle handle(std::move(master));
  bool registered = false;
  {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!shutdown_started_ && !masters_.contains(model_id)) {
      const bool is_default = default_model_.empty();
      masters_.emplace(std::move(model_id), std::move(handle));
      if (is_default) {
        default_model_ = masters_.begin()->first;
      }
      registered = true;
    }
  }
  return registered;
}

bool MasterManager::register_external_master(const std::string& model_id,
                                             Master* master) {
  if (model_id.empty() || master == nullptr) {
    return false;
  }

  MasterHandle handle = make_external_master_handle(master);
  bool registered = false;
  {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!shutdown_started_ && !masters_.contains(model_id)) {
      const bool is_default = default_model_.empty();
      masters_.emplace(model_id, std::move(handle));
      if (is_default) {
        default_model_ = model_id;
      }
      registered = true;
    }
  }
  return registered;
}

bool MasterManager::has_master(const std::string& model_id) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return masters_.contains(model_id);
}

MasterManager::MasterHandle MasterManager::find_master(
    const std::string& model_id) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  auto it = masters_.find(model_id);
  if (it == masters_.end()) {
    return nullptr;
  }
  return it->second;
}

MasterManager::MasterHandle MasterManager::default_master() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  if (default_model_.empty()) {
    return nullptr;
  }
  auto it = masters_.find(default_model_);
  if (it == masters_.end()) {
    return nullptr;
  }
  return it->second;
}

std::vector<std::string> MasterManager::model_ids() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  std::vector<std::string> ids;
  ids.reserve(masters_.size());
  for (const auto& entry : masters_) {
    ids.push_back(entry.first);
  }
  return ids;
}

bool MasterManager::set_default_model(const std::string& model_id) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (!masters_.contains(model_id)) {
    return false;
  }
  default_model_ = model_id;
  return true;
}

void MasterManager::shutdown() {
  std::vector<MasterHandle> masters;
  {
    std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
    {
      std::unique_lock<std::shared_mutex> lock(mutex_);
      if (shutdown_started_) {
        return;
      }
      shutdown_started_ = true;
      masters.reserve(masters_.size());
      for (const auto& entry : masters_) {
        masters.push_back(entry.second);
      }
      masters_.clear();
      default_model_.clear();
    }
  }
}

}  // namespace xllm
