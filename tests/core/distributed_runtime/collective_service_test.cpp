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

#include "core/distributed_runtime/collective_service.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(CollectiveServiceTest, RendezvousNeedsNoDeviceOrCommunicator) {
  // Address discovery must work before device/communicator initialization.
  CollectiveService service(/*total_num=*/2);
  proto::AddressInfo worker;
  proto::CommUniqueIdList response;
  worker.set_global_rank(1);
  worker.set_address("127.0.0.1:9001");
  service.Sync(nullptr, &worker, &response, nullptr);
  EXPECT_EQ(response.comm_unique_ids_size(), 0);

  worker.set_global_rank(0);
  worker.set_address("127.0.0.1:9000");
  service.Sync(nullptr, &worker, &response, nullptr);
  EXPECT_EQ(response.comm_unique_ids_size(), 0);

  const auto addresses = service.wait();
  ASSERT_EQ(addresses.size(), 2);
  EXPECT_EQ(addresses.at(0), "127.0.0.1:9000");
  EXPECT_EQ(addresses.at(1), "127.0.0.1:9001");
}

}  // namespace
}  // namespace xllm
