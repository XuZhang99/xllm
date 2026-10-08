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

#include "core/distributed_runtime/heartbeat_callback_registry.h"

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <mutex>
#include <thread>

namespace xllm {
namespace {

struct TestRequest {
  int32_t value = 0;
  bool block_callback = false;
};

TEST(HeartbeatCallbackRegistryTest, OlderRegistrationCannotClearNewerOne) {
  detail::HeartbeatCallbackRegistry<TestRequest> registry;
  int32_t older_callback_calls = 0;
  int32_t newer_callback_calls = 0;
  const detail::HeartbeatCallbackRegistry<TestRequest>::RegistrationId
      older_registration = registry.set_callback(
          [&](TestRequest& /*unused*/) { ++older_callback_calls; });
  const detail::HeartbeatCallbackRegistry<TestRequest>::RegistrationId
      newer_registration = registry.set_callback(
          [&](TestRequest& /*unused*/) { ++newer_callback_calls; });

  registry.clear_callback(older_registration);
  TestRequest request;
  registry.invoke(request);
  EXPECT_EQ(older_callback_calls, 0);
  EXPECT_EQ(newer_callback_calls, 1);

  registry.clear_callback(newer_registration);
  registry.invoke(request);
  EXPECT_EQ(older_callback_calls, 0);
  EXPECT_EQ(newer_callback_calls, 1);
}

TEST(HeartbeatCallbackRegistryTest, RestoresPreviousRegistrationWhenCleared) {
  detail::HeartbeatCallbackRegistry<TestRequest> registry;
  int32_t older_callback_calls = 0;
  int32_t newer_callback_calls = 0;
  const detail::HeartbeatCallbackRegistry<TestRequest>::RegistrationId
      older_registration = registry.set_callback(
          [&](TestRequest& /*unused*/) { ++older_callback_calls; });
  const detail::HeartbeatCallbackRegistry<TestRequest>::RegistrationId
      newer_registration = registry.set_callback(
          [&](TestRequest& /*unused*/) { ++newer_callback_calls; });

  registry.clear_callback(newer_registration);
  TestRequest request;
  registry.invoke(request);
  EXPECT_EQ(older_callback_calls, 1);
  EXPECT_EQ(newer_callback_calls, 0);
  registry.clear_callback(older_registration);
}

TEST(HeartbeatCallbackRegistryTest, ClearWaitsForActiveCallback) {
  detail::HeartbeatCallbackRegistry<TestRequest> registry;
  std::condition_variable callback_cv;
  std::mutex callback_mutex;
  bool callback_started = false;
  bool release_callback = false;
  const detail::HeartbeatCallbackRegistry<TestRequest>::RegistrationId
      registration = registry.set_callback([&](TestRequest& request) {
        if (!request.block_callback) {
          request.value = 1;
          return;
        }
        std::unique_lock<std::mutex> lock(callback_mutex);
        callback_started = true;
        callback_cv.notify_all();
        callback_cv.wait(lock, [&] { return release_callback; });
        request.value = 42;
      });

  TestRequest request;
  request.block_callback = true;
  std::thread callback_thread([&] { registry.invoke(request); });
  {
    std::unique_lock<std::mutex> lock(callback_mutex);
    callback_cv.wait(lock, [&] { return callback_started; });
  }

  std::future<void> clear_finished = std::async(
      std::launch::async, [&] { registry.clear_callback(registration); });

  const auto clear_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(1);
  TestRequest after_clear;
  bool callback_removed = false;
  while (std::chrono::steady_clock::now() < clear_deadline) {
    registry.invoke(after_clear);
    if (after_clear.value == 0) {
      callback_removed = true;
      break;
    }
    after_clear.value = 0;
    std::this_thread::yield();
  }
  EXPECT_TRUE(callback_removed);
  EXPECT_EQ(clear_finished.wait_for(std::chrono::milliseconds(0)),
            std::future_status::timeout);

  {
    std::lock_guard<std::mutex> lock(callback_mutex);
    release_callback = true;
  }
  callback_cv.notify_all();
  callback_thread.join();
  EXPECT_EQ(clear_finished.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  clear_finished.get();
  EXPECT_EQ(request.value, 42);

  TestRequest after_clear_finished;
  registry.invoke(after_clear_finished);
  EXPECT_EQ(after_clear_finished.value, 0);
}

}  // namespace
}  // namespace xllm
