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

#include "api_service/control_service_impl.h"

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <butil/iobuf.h>
#include <glog/logging.h>
#include <google/protobuf/arena.h>
#include <json2pb/json_to_pb.h>
#include <json2pb/pb_to_json.h>

#include <filesystem>
#include <utility>
#include <vector>

#include "core/common/options.h"
#include "core/common/types.h"
#include "core/distributed_runtime/master_manager.h"
#include "core/framework/config/profile_config.h"

namespace xllm {

namespace {

bool parse_fork_master_request(const proto::MasterInfos* request,
                               Options& options) {
  if (!std::filesystem::exists(request->model_path())) {
    LOG(ERROR) << "Model path " << request->model_path() << " does not exist.";
    return false;
  }

  std::filesystem::path model_path =
      std::filesystem::path(request->model_path()).lexically_normal();
  std::string model_id;
  if (model_path.has_filename()) {
    model_id = model_path.filename().string();
  } else {
    model_id = model_path.parent_path().filename().string();
  }
  options.model_id() = model_id;
  options.master_node_addr() = request->master_node_addr();
  options.model_path() = request->model_path();
  options.master_status() = MasterStatus(request->master_status());

  // Parse nnodes and dp_size (tp_size = nnodes / dp_size, computed by engine)
  if (request->nnodes() > 0) {
    options.nnodes() = request->nnodes();
  }
  if (request->dp_size() > 0) {
    options.dp_size() = request->dp_size();
  }

  return true;
}

}  // namespace

ControlServiceImpl::ControlServiceImpl(
    std::shared_ptr<MasterManager> master_manager)
    : master_manager_(std::move(master_manager)) {
  CHECK(master_manager_ != nullptr);
}

bool ControlServiceImpl::do_fork_master(const proto::MasterInfos& request,
                                        std::string* error_message) {
  Options master_options;
  if (!parse_fork_master_request(&request, master_options)) {
    *error_message = "Failed to parse fork master request";
    return false;
  }
  return master_manager_->fork_master(master_options, error_message);
}

void ControlServiceImpl::fork_master(
    ::google::protobuf::RpcController* controller,
    const proto::MasterInfos* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  auto* ctrl = static_cast<brpc::Controller*>(controller);
  std::string error_message;
  const bool ok = do_fork_master(*request, &error_message);
  response->set_ok(ok);
  if (!ok) {
    LOG(ERROR) << "fork_master failed: " << error_message;
    ctrl->SetFailed(error_message);
  }
}

void ControlServiceImpl::fork_master_http(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::MasterInfos>(arena);

  auto* ctrl = static_cast<brpc::Controller*>(controller);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  const bool st =
      json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  std::string error_message;
  if (!do_fork_master(*req_pb, &error_message)) {
    LOG(ERROR) << "fork_master failed: " << error_message;
    ctrl->SetFailed(error_message);
  }
}

bool ControlServiceImpl::do_sleep(const proto::MasterInfos& request,
                                  std::string* error_message) {
  const MasterStatus req_master_status(request.master_status());
  return master_manager_->sleep(
      request.model_id(), req_master_status, error_message);
}

void ControlServiceImpl::sleep(::google::protobuf::RpcController* controller,
                               const proto::MasterInfos* request,
                               proto::Status* response,
                               ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  auto* ctrl = static_cast<brpc::Controller*>(controller);
  std::string error_message;
  const bool ok = do_sleep(*request, &error_message);
  response->set_ok(ok);
  if (!ok) {
    ctrl->SetFailed(error_message);
  }
}

void ControlServiceImpl::sleep_http(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::MasterInfos>(arena);

  auto* ctrl = static_cast<brpc::Controller*>(controller);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  const bool st =
      json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  std::string error_message;
  if (!do_sleep(*req_pb, &error_message)) {
    ctrl->SetFailed(error_message);
  }
  // Success: return HTTP 200 with empty body
}

bool ControlServiceImpl::do_wakeup(const proto::MasterInfos& request,
                                   std::string* error_message) {
  // Parse remote weight transfer parameters before handing over lifecycle
  // control to the manager.
  WakeupOptions wakeup_options;
  if (request.remote_addrs_size() > 0) {
    wakeup_options.remote_addrs.assign(request.remote_addrs().begin(),
                                       request.remote_addrs().end());
    if (request.src_weight_segments_size() > 0) {
      wakeup_options.src_weight_segments.reserve(
          request.src_weight_segments_size());
      for (const auto& seg_list : request.src_weight_segments()) {
        std::vector<WeightSegment> segments;
        segments.reserve(seg_list.segments_size());
        for (const auto& proto_seg : seg_list.segments()) {
          segments.emplace_back(proto_seg.offset(), proto_seg.size());
        }
        wakeup_options.src_weight_segments.emplace_back(std::move(segments));
      }
    }
  }
  return master_manager_->wakeup(
      request.model_id(), wakeup_options, error_message);
}

void ControlServiceImpl::wakeup(::google::protobuf::RpcController* controller,
                                const proto::MasterInfos* request,
                                proto::Status* response,
                                ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  auto* ctrl = static_cast<brpc::Controller*>(controller);
  std::string error_message;
  const bool ok = do_wakeup(*request, &error_message);
  response->set_ok(ok);
  if (!ok) {
    ctrl->SetFailed(error_message);
  }
}

void ControlServiceImpl::wakeup_http(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::MasterInfos>(arena);

  auto* ctrl = static_cast<brpc::Controller*>(controller);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  const bool st =
      json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  std::string error_message;
  if (!do_wakeup(*req_pb, &error_message)) {
    ctrl->SetFailed(error_message);
  }
  // Success: return HTTP 200 with empty body
}

void ControlServiceImpl::start_profile_http(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  auto* ctrl = static_cast<brpc::Controller*>(controller);

  if (!ProfileConfig::get_instance().enable_online_profile()) {
    LOG(ERROR) << "Profiling is disabled. Start the server with "
                  "--enable_online_profile=true to use /start_profile.";
    ctrl->SetFailed(
        "Profiling is disabled. Start the server with "
        "--enable_online_profile=true.");
    return;
  }
  LOG(INFO) << "Starting profiler.";
  std::string error_message;
  if (!master_manager_->start_profile(&error_message)) {
    LOG(ERROR) << error_message;
    ctrl->SetFailed(error_message);
    return;
  }
  LOG(INFO) << "Profiler started.";
  // Success: return HTTP 200 with empty body
}

void ControlServiceImpl::stop_profile_http(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  auto* ctrl = static_cast<brpc::Controller*>(controller);

  if (!ProfileConfig::get_instance().enable_online_profile()) {
    LOG(ERROR) << "Profiling is disabled. Start the server with "
                  "--enable_online_profile=true to use /stop_profile.";
    ctrl->SetFailed(
        "Profiling is disabled. Start the server with "
        "--enable_online_profile=true.");
    return;
  }
  LOG(INFO) << "Stopping profiler.";
  std::string error_message;
  if (!master_manager_->stop_profile(&error_message)) {
    LOG(ERROR) << error_message;
    ctrl->SetFailed(error_message);
    return;
  }
  LOG(INFO) << "Profiler stopped.";
  // Success: return HTTP 200 with empty body
}

void ControlServiceImpl::link_p2p(::google::protobuf::RpcController* controller,
                                  const proto::P2PLinkRequest* request,
                                  proto::Status* response,
                                  ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  std::string error_message;
  const bool status = master_manager_->link_p2p(
      request->model_id(),
      {request->remote_addrs().begin(), request->remote_addrs().end()},
      &error_message);
  if (!status) {
    LOG(ERROR) << error_message;
  }
  response->set_ok(status);
}

void ControlServiceImpl::link_p2p_http(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::P2PLinkRequest>(arena);
  auto resp_pb = google::protobuf::Arena::CreateMessage<proto::Status>(arena);

  auto* ctrl = static_cast<brpc::Controller*>(controller);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  const bool st =
      json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  if (!master_manager_->has_master(req_pb->model_id())) {
    LOG(ERROR) << "Master for model " << req_pb->model_id() << " not found";
    ctrl->SetFailed("Master for model not found");
    return;
  }

  std::string error_message;
  const bool status = master_manager_->link_p2p(
      req_pb->model_id(),
      {req_pb->remote_addrs().begin(), req_pb->remote_addrs().end()},
      &error_message);
  if (!status) {
    LOG(ERROR) << error_message;
  }
  resp_pb->set_ok(status);

  json2pb::Pb2JsonOptions json_options;
  json_options.bytes_to_base64 = false;
  std::string err_msg;
  butil::IOBufAsZeroCopyOutputStream json_output(&ctrl->response_attachment());
  if (!json2pb::ProtoMessageToJson(
          *resp_pb, &json_output, json_options, &err_msg)) {
    LOG(ERROR) << "proto to json failed: " << err_msg;
    return;
  }
}

void ControlServiceImpl::unlink_p2p(
    ::google::protobuf::RpcController* controller,
    const proto::P2PLinkRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  std::string error_message;
  const bool status = master_manager_->unlink_p2p(
      request->model_id(),
      {request->remote_addrs().begin(), request->remote_addrs().end()},
      &error_message);
  if (!status) {
    LOG(ERROR) << error_message;
  }
  response->set_ok(status);
}

void ControlServiceImpl::unlink_p2p_http(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  auto arena = response->GetArena();
  auto req_pb =
      google::protobuf::Arena::CreateMessage<proto::P2PLinkRequest>(arena);
  auto resp_pb = google::protobuf::Arena::CreateMessage<proto::Status>(arena);

  auto* ctrl = static_cast<brpc::Controller*>(controller);

  std::string error;
  json2pb::Json2PbOptions options;
  butil::IOBuf& buf = ctrl->request_attachment();
  butil::IOBufAsZeroCopyInputStream iobuf_stream(buf);
  const bool st =
      json2pb::JsonToProtoMessage(&iobuf_stream, req_pb, options, &error);
  if (!st) {
    ctrl->SetFailed(error);
    LOG(ERROR) << "parse json to proto failed: " << error;
    return;
  }

  if (!master_manager_->has_master(req_pb->model_id())) {
    LOG(ERROR) << "Master for model " << req_pb->model_id() << " not found";
    ctrl->SetFailed("Master for model not found");
    return;
  }

  std::string error_message;
  const bool status = master_manager_->unlink_p2p(
      req_pb->model_id(),
      {req_pb->remote_addrs().begin(), req_pb->remote_addrs().end()},
      &error_message);
  if (!status) {
    LOG(ERROR) << error_message;
  }
  resp_pb->set_ok(status);

  json2pb::Pb2JsonOptions json_options;
  json_options.bytes_to_base64 = false;
  std::string err_msg;
  butil::IOBufAsZeroCopyOutputStream json_output(&ctrl->response_attachment());
  if (!json2pb::ProtoMessageToJson(
          *resp_pb, &json_output, json_options, &err_msg)) {
    LOG(ERROR) << "proto to json failed: " << err_msg;
    return;
  }
}

}  // namespace xllm
