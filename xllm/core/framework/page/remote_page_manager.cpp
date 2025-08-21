#include "remote_page_manager.h"

#include <brpc/controller.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <chrono>
#include <memory>
#include <optional>
#include <utility>

namespace xllm {
RemotePageManger::RemotePageManger(int32_t global_rank,
                                   const std::string& server_address,
                                   const torch::Device& d)
    : global_rank_(global_rank), device_(d) {
  // Initialize brpc channel
  options_.connection_type = "pooled";
  options_.timeout_ms = -1;
  options_.connect_timeout_ms = -1;
  options_.max_retry = 3;
  if (channel_.Init(server_address.c_str(), "", &options_) != 0) {
    LOG(ERROR) << "Failed to initialize brpc channel";
    return;
  }
  // Initialize stub
  stub_.reset(new proto::DistributePageManager_Stub(&channel_));

  wait_for_server_ready(server_address);
}

bool RemotePageManger::wait_for_server_ready(
    const std::string& server_address) {
  proto::Status req;
  proto::Status resp;

  // Retry until server initialize ready
  int try_count = 0;
  brpc::Controller cntl;
  while (try_count < FLAGS_max_connect_count) {
    cntl.Reset();
    stub_->Hello(&cntl, &req, &resp, nullptr);
    if (cntl.Failed() || !resp.ok()) {
      std::this_thread::sleep_for(
          std::chrono::seconds(FLAGS_sleep_time_second));
    } else {
      LOG(INFO) << "RemotePageManger Hello connected, server_address: "
                << server_address << ", global_rank_: " << global_rank_;
      break;
    }

    try_count++;
  }

  if (try_count >= FLAGS_max_connect_count) {
    LOG(ERROR) << "RemotePageManger Hello method failed, global_rank_ is "
               << global_rank_ << ", error: " << cntl.ErrorText();
    return false;
  }

  return true;
}

bool RemotePageManger::allocate(int32_t& seq_id, size_t num_tokens) {
  proto::AllocatePagesRequest req;
  req.set_seq_id(seq_id);
  req.set_num_tokens(num_tokens);
  proto::AllocatePagesResponse resp;
  brpc::Controller cntl;
  stub_->Allocate(&cntl, &req, &resp, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "Allocate method failed: " << cntl.ErrorText();
    return false;
  }
  seq_id = resp.seq_id();
  return resp.success();
}

void RemotePageManger::deallocate(int32_t seq_id) {
  proto::SeqId req;
  req.set_seq_id(seq_id);
  proto::Empty resp;
  brpc::Controller cntl;
  stub_->Deallocate(&cntl, &req, &resp, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "Deallocate method failed: " << cntl.ErrorText();
  }
}

void RemotePageManger::cache(int32_t seq_id) {
  proto::SeqId req;
  req.set_seq_id(seq_id);
  proto::Empty resp;
  brpc::Controller cntl;
  stub_->Cache(&cntl, &req, &resp, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "Cache method failed: " << cntl.ErrorText();
  }
}

folly::SemiFuture<bool> RemotePageManger::allocate_async(int32_t& seq_id,
                                                         size_t num_tokens) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();

  threadpool_.schedule(
      [this, seq_id, num_tokens, promise = std::move(promise)]() mutable {
        proto::AllocatePagesRequest req;
        req.set_seq_id(seq_id);
        req.set_num_tokens(num_tokens);
        proto::AllocatePagesResponse resp;
        brpc::Controller cntl;
        stub_->Allocate(&cntl, &req, &resp, nullptr);
        if (cntl.Failed()) {
          LOG(ERROR) << "Allocate method failed: " << cntl.ErrorText();
        }
        seq_id = resp.seq_id();
        promise.setValue(resp.success());
      });
  return future;
}

folly::SemiFuture<folly::Unit> RemotePageManger::deallocate_async(
    int32_t seq_id) {
  folly::Promise<folly::Unit> promise;
  auto future = promise.getSemiFuture();

  threadpool_.schedule([this, seq_id, promise = std::move(promise)]() mutable {
    proto::SeqId req;
    req.set_seq_id(seq_id);
    proto::Empty resp;
    brpc::Controller cntl;
    stub_->Deallocate(&cntl, &req, &resp, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "Deallocate method failed: " << cntl.ErrorText();
    }
    promise.setValue();
  });
  return future;
}

folly::SemiFuture<folly::Unit> RemotePageManger::cache_async(int32_t seq_id) {
  folly::Promise<folly::Unit> promise;
  auto future = promise.getSemiFuture();

  threadpool_.schedule([this, seq_id, promise = std::move(promise)]() mutable {
    proto::SeqId req;
    req.set_seq_id(seq_id);
    proto::Empty resp;
    brpc::Controller cntl;
    stub_->Cache(&cntl, &req, &resp, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "Cache method failed: " << cntl.ErrorText();
    }
    promise.setValue();
  });
  return future;
}

size_t RemotePageManger::num_free_pages_per_layer() const {
  proto::Empty req;
  proto::NumPages resp;
  brpc::Controller cntl;
  stub_->NumFreePagesPerLayer(&cntl, &req, &resp, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "NumFreePagesPerLayer method failed: " << cntl.ErrorText();
  }
  return resp.num_pages();
}

size_t RemotePageManger::num_used_pages_per_layer() const {
  proto::Empty req;
  proto::NumPages resp;
  brpc::Controller cntl;
  stub_->NumUsedPagesPerLayer(&cntl, &req, &resp, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "NumUsedPagesPerLayer method failed: " << cntl.ErrorText();
  }
  return resp.num_pages();
}

double RemotePageManger::kv_cache_utilization() const {
  proto::Empty req;
  proto::KvCacheUtilization resp;
  brpc::Controller cntl;
  stub_->KvCacheUtilization(&cntl, &req, &resp, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "KvCacheUtilization method failed: " << cntl.ErrorText();
  }
  return resp.kv_cache_utilization();
}
}  // namespace xllm