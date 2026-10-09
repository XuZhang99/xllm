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

#include "core/distributed_runtime/distributed_worker_manager.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace xllm {
namespace {

class RecordingWorkerClient final : public WorkerClient {
 public:
  bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                    const std::vector<std::string>& addrs,
                    const std::vector<uint16_t>& ports) override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++link_cluster_calls_;
    cluster_ids_ = cluster_ids;
    addrs_ = addrs;
    ports_ = ports;
    return link_cluster_result_;
  }

  bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                      const std::vector<std::string>& addrs,
                      const std::vector<uint16_t>& ports) override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++unlink_cluster_calls_;
    cluster_ids_ = cluster_ids;
    addrs_ = addrs;
    ports_ = ports;
    return unlink_cluster_result_;
  }

  bool link_p2p(const std::string& remote_addr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++link_p2p_calls_;
    remote_addr_ = remote_addr;
    return link_p2p_result_;
  }

  bool unlink_p2p(const std::string& remote_addr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++unlink_p2p_calls_;
    remote_addr_ = remote_addr;
    return unlink_p2p_result_;
  }

  bool link_cluster_result_ = true;
  bool unlink_cluster_result_ = true;
  bool link_p2p_result_ = true;
  bool unlink_p2p_result_ = true;
  int32_t link_cluster_calls_ = 0;
  int32_t unlink_cluster_calls_ = 0;
  int32_t link_p2p_calls_ = 0;
  int32_t unlink_p2p_calls_ = 0;
  std::vector<uint64_t> cluster_ids_;
  std::vector<std::string> addrs_;
  std::vector<uint16_t> ports_;
  std::string remote_addr_;

 private:
  std::mutex mutex_;
};

}  // namespace

class DistributedWorkerManagerTest : public ::testing::Test {
 protected:
  std::unique_ptr<DistributedWorkerManager> make_manager(
      std::vector<std::shared_ptr<RecordingWorkerClient>> clients) {
    std::vector<std::shared_ptr<WorkerClient>> worker_clients;
    worker_clients.reserve(clients.size());
    for (auto& client : clients) {
      worker_clients.emplace_back(std::move(client));
    }
    return std::unique_ptr<DistributedWorkerManager>(
        new DistributedWorkerManager(std::move(worker_clients)));
  }

  static std::vector<uint64_t> cluster_ids() { return {11, 22}; }
  static std::vector<std::string> addrs() {
    return {"127.0.0.1:1001", "127.0.0.1:1002"};
  }
  static std::vector<uint16_t> ports() { return {1001, 1002}; }
};

TEST_F(DistributedWorkerManagerTest, LinksAndUnlinksAllWorkers) {
  auto first = std::make_shared<RecordingWorkerClient>();
  auto second = std::make_shared<RecordingWorkerClient>();
  auto manager = make_manager({first, second});
  EXPECT_FALSE(manager->link_threadpool_);

  EXPECT_TRUE(manager->link_cluster(cluster_ids(),
                                    addrs(),
                                    ports(),
                                    /*src_dp_size=*/1));
  EXPECT_TRUE(manager->link_threadpool_);
  EXPECT_TRUE(manager->unlink_cluster(cluster_ids(),
                                      addrs(),
                                      ports(),
                                      /*src_dp_size=*/1));
  EXPECT_TRUE(manager->link_p2p({"p2p-0", "p2p-1"}));
  EXPECT_TRUE(manager->unlink_p2p({"p2p-0", "p2p-1"}));

  for (const auto& client : {first, second}) {
    EXPECT_EQ(client->link_cluster_calls_, 1);
    EXPECT_EQ(client->unlink_cluster_calls_, 1);
    EXPECT_EQ(client->cluster_ids_, cluster_ids());
    EXPECT_EQ(client->addrs_, addrs());
    EXPECT_EQ(client->ports_, ports());
    EXPECT_EQ(client->link_p2p_calls_, 1);
    EXPECT_EQ(client->unlink_p2p_calls_, 1);
  }
  EXPECT_EQ(first->remote_addr_, "p2p-0");
  EXPECT_EQ(second->remote_addr_, "p2p-1");
}

TEST_F(DistributedWorkerManagerTest, ReportsWorkerFailures) {
  auto first = std::make_shared<RecordingWorkerClient>();
  auto second = std::make_shared<RecordingWorkerClient>();
  second->link_cluster_result_ = false;
  second->unlink_cluster_result_ = false;
  second->link_p2p_result_ = false;
  second->unlink_p2p_result_ = false;
  auto manager = make_manager({first, second});

  EXPECT_FALSE(manager->link_cluster(cluster_ids(),
                                     addrs(),
                                     ports(),
                                     /*src_dp_size=*/1));
  EXPECT_FALSE(manager->unlink_cluster(cluster_ids(),
                                       addrs(),
                                       ports(),
                                       /*src_dp_size=*/1));
  EXPECT_FALSE(manager->link_p2p({"p2p-0", "p2p-1"}));
  EXPECT_FALSE(manager->unlink_p2p({"p2p-0", "p2p-1"}));
  EXPECT_EQ(first->link_cluster_calls_, 1);
  EXPECT_EQ(second->link_cluster_calls_, 1);
  EXPECT_EQ(first->link_p2p_calls_, 1);
  EXPECT_EQ(second->link_p2p_calls_, 1);
  EXPECT_EQ(first->unlink_cluster_calls_, 1);
  EXPECT_EQ(second->unlink_cluster_calls_, 1);
  EXPECT_EQ(first->unlink_p2p_calls_, 1);
  EXPECT_EQ(second->unlink_p2p_calls_, 1);
}

TEST_F(DistributedWorkerManagerTest, RejectsInvalidTopologyWithoutDispatch) {
  auto client = std::make_shared<RecordingWorkerClient>();
  auto manager = make_manager({client});
  EXPECT_FALSE(manager->link_threadpool_);

  const auto expect_rejected = [&manager](
                                   const std::vector<uint64_t>& ids,
                                   const std::vector<std::string>& addresses,
                                   const std::vector<uint16_t>& source_ports,
                                   int32_t dp_size,
                                   int32_t kv_split_size) {
    EXPECT_FALSE(manager->link_cluster(
        ids, addresses, source_ports, dp_size, kv_split_size));
    EXPECT_FALSE(manager->unlink_cluster(
        ids, addresses, source_ports, dp_size, kv_split_size));
  };
  expect_rejected({}, {}, {}, /*dp_size=*/1, /*kv_split_size=*/1);
  expect_rejected(cluster_ids(),
                  addrs(),
                  ports(),
                  /*dp_size=*/0,
                  /*kv_split_size=*/1);
  expect_rejected(cluster_ids(),
                  addrs(),
                  ports(),
                  /*dp_size=*/1,
                  /*kv_split_size=*/0);
  expect_rejected(cluster_ids(),
                  addrs(),
                  ports(),
                  /*dp_size=*/3,
                  /*kv_split_size=*/1);
  expect_rejected(cluster_ids(),
                  addrs(),
                  ports(),
                  /*dp_size=*/1,
                  /*kv_split_size=*/3);
  expect_rejected(cluster_ids(),
                  {"one"},
                  ports(),
                  /*dp_size=*/1,
                  /*kv_split_size=*/1);
  expect_rejected(cluster_ids(),
                  addrs(),
                  {1001},
                  /*dp_size=*/1,
                  /*kv_split_size=*/1);
  EXPECT_FALSE(manager->link_p2p({}));
  EXPECT_FALSE(manager->unlink_p2p({"one", "two"}));
  EXPECT_FALSE(manager->link_threadpool_);
  EXPECT_EQ(client->link_cluster_calls_, 0);
  EXPECT_EQ(client->unlink_cluster_calls_, 0);
  EXPECT_EQ(client->link_p2p_calls_, 0);
  EXPECT_EQ(client->unlink_p2p_calls_, 0);
}

TEST_F(DistributedWorkerManagerTest, RejectsOperationsWithoutWorkers) {
  auto manager = make_manager({});

  EXPECT_FALSE(manager->link_cluster(cluster_ids(),
                                     addrs(),
                                     ports(),
                                     /*src_dp_size=*/1));
  EXPECT_FALSE(manager->unlink_cluster(cluster_ids(),
                                       addrs(),
                                       ports(),
                                       /*src_dp_size=*/1));
  EXPECT_FALSE(manager->link_p2p({}));
  EXPECT_FALSE(manager->unlink_p2p({}));
}

}  // namespace xllm
