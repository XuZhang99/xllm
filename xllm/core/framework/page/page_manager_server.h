#pragma once

#include <brpc/server.h>
#include <folly/futures/Future.h>
#include <torch/torch.h>

#include <string>
#include <thread>

#include "common/macros.h"
#include "page_manager.pb.h"
#include "worker.pb.h"

namespace xllm {
class PageManagerServer {
 public:
  PageManagerServer(int local_worker_idx,
                    const std::string& master_node_addr,
                    std::atomic<bool>& done,
                    const torch::Device& device,
                    const memory::Options& options);
  virtual ~PageManagerServer();

 private:
  DISALLOW_COPY_AND_ASSIGN(PageManagerServer);

  void create_server(std::atomic<bool>& done,
                     const std::string& master_node_addr,
                     const torch::Device& device,
                     int world_size,
                     int global_rank,
                     int32_t dp_size,
                     int local_rank);

  bool sync_master_node(const std::string& master_node_addr,
                        proto::AddressInfo& addr_info,
                        proto::CommUniqueIdList& uids);

 private:
  std::unique_ptr<std::thread> page_manager_thread_;
};
}  // namespace xllm