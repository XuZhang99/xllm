#pragma once

#include <memory>

#include "common/macros.h"
#include "memory/page_manager.h"
#include "page_manager.pb.h"

namespace xllm {
class PageManagerService : public proto::DistributePageManager {
 public:
  PageManagerService(int32_t global_rank,
                     int32_t world_size,
                     const torch::Device& d);
  ~PageManagerService() = default;

  void set_page_manager(std::unique_ptr<PageManager> page_manager);

  // service functions
  void Hello(::google::protobuf::RpcController* controller,
             const proto::Status* request,
             proto::Status* response,
             ::google::protobuf::Closure* done) override;

  void Allocate(::google::protobuf::RpcController* controller,
                const proto::AllocatePagesRequest* request,
                proto::AllocatePagesResponse* response,
                ::google::protobuf::Closure* done) override;

  void Deallocate(::google::protobuf::RpcController* controller,
                  const proto::SeqId* request,
                  proto::Empty* response,
                  ::google::protobuf::Closure* done) override;

  void Cache(::google::protobuf::RpcController* controller,
             const proto::SeqId* request,
             proto::Empty* response,
             ::google::protobuf::Closure* done) override;

  void NumFreePagesPerLayer(::google::protobuf::RpcController* controller,
                            const proto::Empty* request,
                            proto::NumPages* response,
                            ::google::protobuf::Closure* done) override;

  void NumUsedPagesPerLayer(::google::protobuf::RpcController* controller,
                            const proto::Empty* request,
                            proto::NumPages* response,
                            ::google::protobuf::Closure* done) override;

  void KvCacheUtilization(::google::protobuf::RpcController* controller,
                          const proto::Empty* request,
                          proto::KvCacheUtilization* response,
                          ::google::protobuf::Closure* done) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(PageManagerService);

 private:
  bool initialized_;
  int32_t global_rank_;
  int32_t world_size_;
  torch::Device device_;
  ThreadPool threadpool_{5};
  std::unique_ptr<PageManager> page_manager_;
}

}  // namespace xllm