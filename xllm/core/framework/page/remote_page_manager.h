#pragma once

#include <brpc/channel.h>
#include <folly/futures/Future.h>

namespace xllm {

class RemotePageManger : public PageManagerClient {
 public:
  explicit RemotePageManger(int32_t global_rank,
                            const std::string& server_address,
                            const torch::Device& d);
  virtual ~RemotePageManger() = default;

  bool wait_for_server_ready(const std::string& server_address);

  bool allocate(int32_t& seq_id, size_t num_tokens);
  void deallocate(int32_t seq_id);
  void cache(int32_t seq_id);

  folly::SemiFuture<bool> allocate_async(int32_t& seq_id, size_t num_tokens);
  folly::SemiFuture<folly::Unit> deallocate_async(int32_t seq_id);
  folly::SemiFuture<folly::Unit> cache_async(int32_t seq_id);

  size_t num_free_pages_per_layer() const;
  size_t num_used_pages_per_layer() const;
  double kv_cache_utilization() const;

 private:
  DISALLOW_COPY_AND_ASSIGN(RemotePageManger);

 private:
  int32_t global_rank_;

  // brpc connection resource
  brpc::Channel channel_;
  brpc::ChannelOptions options_;
  std::unique_ptr<proto::DistributePageManager_Stub> stub_;

  ThreadPool threadpool_;
  const torch::Device device_;
}
}  // namespace xllm