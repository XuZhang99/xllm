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

#include <glog/logging.h>

#include <string>
#include <utility>

#include "api_service/control_request_utils.h"
#include "core/common/options.h"
#include "core/distributed_runtime/master_manager.h"
#include "core/framework/config/profile_config.h"

namespace xllm {

ControlServiceImpl::ControlServiceImpl(
    std::shared_ptr<MasterManager> master_manager)
    : master_manager_(std::move(master_manager)) {
  CHECK(master_manager_ != nullptr);
}

Status ControlServiceImpl::fork_master(const proto::MasterInfos& request) {
  Options master_options;
  const Status status =
      api_service::parse_fork_master_request(request, master_options);
  if (!status.ok()) {
    LOG(ERROR) << "fork_master failed: " << status.message();
    return status;
  }
  std::string error_message;
  if (!master_manager_->fork_master(master_options, &error_message)) {
    LOG(ERROR) << "fork_master failed: " << error_message;
    return Status(StatusCode::UNKNOWN, std::move(error_message));
  }
  return Status();
}

Status ControlServiceImpl::sleep(const proto::MasterInfos& request) {
  const MasterStatus master_status(request.master_status());
  std::string error_message;
  if (!master_manager_->sleep(
          request.model_id(), master_status, &error_message)) {
    return Status(StatusCode::UNKNOWN, std::move(error_message));
  }
  return Status();
}

Status ControlServiceImpl::wakeup(const proto::MasterInfos& request) {
  const WakeupOptions wakeup_options =
      api_service::parse_wakeup_options(request);
  std::string error_message;
  if (!master_manager_->wakeup(
          request.model_id(), wakeup_options, &error_message)) {
    return Status(StatusCode::UNKNOWN, std::move(error_message));
  }
  return Status();
}

Status ControlServiceImpl::start_profile() {
  if (!ProfileConfig::get_instance().enable_online_profile()) {
    LOG(ERROR) << "Profiling is disabled. Start the server with "
                  "--enable_online_profile=true to use /start_profile.";
    return Status(StatusCode::UNAVAILABLE,
                  "Profiling is disabled. Start the server with "
                  "--enable_online_profile=true.");
  }
  LOG(INFO) << "Starting profiler.";
  std::string error_message;
  if (!master_manager_->start_profile(&error_message)) {
    LOG(ERROR) << error_message;
    return Status(StatusCode::UNKNOWN, std::move(error_message));
  }
  LOG(INFO) << "Profiler started.";
  return Status();
}

Status ControlServiceImpl::stop_profile() {
  if (!ProfileConfig::get_instance().enable_online_profile()) {
    LOG(ERROR) << "Profiling is disabled. Start the server with "
                  "--enable_online_profile=true to use /stop_profile.";
    return Status(StatusCode::UNAVAILABLE,
                  "Profiling is disabled. Start the server with "
                  "--enable_online_profile=true.");
  }
  LOG(INFO) << "Stopping profiler.";
  std::string error_message;
  if (!master_manager_->stop_profile(&error_message)) {
    LOG(ERROR) << error_message;
    return Status(StatusCode::UNKNOWN, std::move(error_message));
  }
  LOG(INFO) << "Profiler stopped.";
  return Status();
}

Status ControlServiceImpl::link_p2p(const proto::P2PLinkRequest& request) {
  if (!master_manager_->has_master(request.model_id())) {
    LOG(ERROR) << "Master for model " << request.model_id() << " not found";
    return Status(StatusCode::NOT_FOUND, "Master for model not found");
  }
  std::string error_message;
  if (!master_manager_->link_p2p(
          request.model_id(),
          {request.remote_addrs().begin(), request.remote_addrs().end()},
          &error_message)) {
    LOG(ERROR) << error_message;
    return Status(StatusCode::UNKNOWN, std::move(error_message));
  }
  return Status();
}

Status ControlServiceImpl::unlink_p2p(const proto::P2PLinkRequest& request) {
  if (!master_manager_->has_master(request.model_id())) {
    LOG(ERROR) << "Master for model " << request.model_id() << " not found";
    return Status(StatusCode::NOT_FOUND, "Master for model not found");
  }
  std::string error_message;
  if (!master_manager_->unlink_p2p(
          request.model_id(),
          {request.remote_addrs().begin(), request.remote_addrs().end()},
          &error_message)) {
    LOG(ERROR) << error_message;
    return Status(StatusCode::UNKNOWN, std::move(error_message));
  }
  return Status();
}

}  // namespace xllm
