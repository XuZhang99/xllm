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

#include <glog/logging.h>

#include "core/distributed_runtime/llm_master.h"
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

bool MasterManager::sleep(const std::string& model_id,
                          MasterStatus master_status,
                          std::string* error_message) {
  if (master_status != MasterStatus::LIGHT_SLEEP &&
      master_status != MasterStatus::DEEP_SLEEP) {
    LOG(ERROR) << "Invalid sleep status: " << master_status.to_proto();
    if (error_message != nullptr) {
      *error_message = "Invalid sleep status";
    }
    return false;
  }

  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
  const MasterHandle master = find_master(model_id);
  if (master == nullptr) {
    LOG(ERROR) << "Master for model " << model_id << " not found";
    if (error_message != nullptr) {
      *error_message = "Master for model not found";
    }
    return false;
  }

  auto* llm_master = dynamic_cast<LLMMaster*>(master.get());
  if (llm_master == nullptr) {
    if (error_message != nullptr) {
      *error_message = "Sleep is only supported for LLM masters";
    }
    return false;
  }
  if (llm_master->is_sleeping()) {
    LOG(INFO) << "Master for model " << model_id << " is already sleeping";
    if (error_message != nullptr) {
      *error_message = "Master for model is already sleeping";
    }
    return false;
  }

  RateLimiter* rate_limiter = master->get_rate_limiter();
  if (!rate_limiter->try_set_sleeping()) {
    const int32_t num_requests =
        rate_limiter->get_num_concurrent_requests();
    LOG(ERROR) << "Cannot sleep model " << model_id << " with "
               << num_requests << " in-flight requests";
    if (error_message != nullptr) {
      *error_message = "Cannot sleep model with in-flight requests";
    }
    return false;
  }

  const MasterStatus previous_status = llm_master->get_master_status();
  llm_master->set_master_status(master_status);
  if (llm_master->sleep()) {
    return true;
  }

  // Keep the sleep marker after a failed engine transition. The allocator can
  // have released only part of the resources, so admitting requests here
  // could send them to an incompletely sleeping engine. A later wakeup can
  // retry the transition using the requested sleep mode.
  LOG(ERROR) << "Failed to sleep model " << model_id << " from status "
             << previous_status.to_proto()
             << "; keeping the model unavailable for requests";
  if (error_message != nullptr) {
    *error_message = "Failed to sleep model";
  }
  return false;
}

bool MasterManager::wakeup(const std::string& model_id,
                           const WakeupOptions& options,
                           std::string* error_message) {
  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);
  const MasterHandle master = find_master(model_id);
  if (master == nullptr) {
    LOG(ERROR) << "Master for model " << model_id << " not found";
    if (error_message != nullptr) {
      *error_message = "Master for model not found";
    }
    return false;
  }

  auto* llm_master = dynamic_cast<LLMMaster*>(master.get());
  if (llm_master == nullptr) {
    if (error_message != nullptr) {
      *error_message = "Wakeup is only supported for LLM masters";
    }
    return false;
  }
  if (!llm_master->is_sleeping()) {
    LOG(INFO) << "Master for model " << model_id << " is already awake";
    if (error_message != nullptr) {
      *error_message = "Master for model is already awake";
    }
    return false;
  }

  RateLimiter* rate_limiter = master->get_rate_limiter();
  if (!rate_limiter->is_sleeping()) {
    LOG(ERROR) << "Cannot wakeup model " << model_id
               << " because its rate limiter is not sleeping";
    if (error_message != nullptr) {
      *error_message = "Cannot wakeup model with an inconsistent rate limiter";
    }
    return false;
  }

  const bool has_remote_weights = !options.remote_addrs.empty();
  const bool wakeup_succeeded = has_remote_weights
                                    ? llm_master->wakeup(options)
                                    : llm_master->wakeup();
  if (!wakeup_succeeded) {
    LOG(ERROR) << "Failed to wakeup model " << model_id
               << (has_remote_weights ? " with remote weight transfer" : "");
    if (error_message != nullptr) {
      *error_message = has_remote_weights
                           ? "Failed to wakeup model with remote weight transfer"
                           : "Failed to wakeup model";
    }
    return false;
  }

  if (!rate_limiter->try_wakeup()) {
    LOG(ERROR) << "Failed to restore rate limiter for model " << model_id;
    if (error_message != nullptr) {
      *error_message = "Failed to restore rate limiter";
    }
    return false;
  }

  llm_master->set_master_status(MasterStatus::WAKEUP);
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
