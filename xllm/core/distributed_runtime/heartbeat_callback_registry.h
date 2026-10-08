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

#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "util/scope_guard.h"

namespace xllm::detail {

template <typename Request>
class HeartbeatCallbackRegistry final {
 public:
  using RegistrationId = std::uint64_t;
  using Callback = std::function<void(Request&)>;

  RegistrationId set_callback(Callback callback) {
    if (!callback) {
      return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    do {
      ++last_registration_id_;
    } while (last_registration_id_ == 0 ||
             callbacks_.contains(last_registration_id_) ||
             active_callbacks_.contains(last_registration_id_));
    callbacks_.emplace(last_registration_id_, std::move(callback));
    registration_order_.emplace_back(last_registration_id_);
    current_registration_id_ = last_registration_id_;
    return current_registration_id_;
  }

  void clear_callback(RegistrationId registration_id) {
    if (registration_id == 0) {
      return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    callbacks_.erase(registration_id);
    registration_order_.erase(std::remove(registration_order_.begin(),
                                          registration_order_.end(),
                                          registration_id),
                              registration_order_.end());
    if (current_registration_id_ == registration_id) {
      current_registration_id_ =
          registration_order_.empty() ? 0 : registration_order_.back();
    }
    callback_cv_.wait(lock, [this, registration_id] {
      return !active_callbacks_.contains(registration_id);
    });
  }

  void invoke(Request& request) {
    Callback callback;
    RegistrationId registration_id = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto current_callback = callbacks_.find(current_registration_id_);
      if (current_callback != callbacks_.end()) {
        callback = current_callback->second;
        registration_id = current_registration_id_;
        ++active_callbacks_[registration_id];
      }
    }
    if (!callback) {
      return;
    }

    xllm::ScopeGuard callback_guard([this, registration_id] {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        auto active_callback = active_callbacks_.find(registration_id);
        if (active_callback != active_callbacks_.end() &&
            --active_callback->second == 0) {
          active_callbacks_.erase(active_callback);
        }
      }
      callback_cv_.notify_all();
    });
    callback(request);
  }

 private:
  std::mutex mutex_;
  std::condition_variable callback_cv_;
  std::unordered_map<RegistrationId, Callback> callbacks_;
  std::vector<RegistrationId> registration_order_;
  RegistrationId current_registration_id_ = 0;
  RegistrationId last_registration_id_ = 0;
  std::unordered_map<RegistrationId, std::size_t> active_callbacks_;
};

}  // namespace xllm::detail
