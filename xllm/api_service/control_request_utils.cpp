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

#include "api_service/control_request_utils.h"

#include <glog/logging.h>

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "core/common/options.h"

namespace xllm::api_service {

Status parse_fork_master_request(const proto::MasterInfos& request,
                                 Options& options) {
  if (!std::filesystem::exists(request.model_path())) {
    LOG(ERROR) << "Model path " << request.model_path() << " does not exist.";
    return {StatusCode::INVALID_ARGUMENT,
            "Failed to parse fork master request"};
  }

  const std::filesystem::path model_path =
      std::filesystem::path(request.model_path()).lexically_normal();
  std::string model_id;
  if (model_path.has_filename()) {
    model_id = model_path.filename().string();
  } else {
    model_id = model_path.parent_path().filename().string();
  }
  options.model_id() = std::move(model_id);
  options.master_node_addr() = request.master_node_addr();
  options.model_path() = request.model_path();
  options.master_status() = MasterStatus(request.master_status());

  // The engine derives tp_size from nnodes / dp_size.
  if (request.nnodes() > 0) {
    options.nnodes() = request.nnodes();
  }
  if (request.dp_size() > 0) {
    options.dp_size() = request.dp_size();
  }

  return {};
}

WakeupOptions parse_wakeup_options(const proto::MasterInfos& request) {
  WakeupOptions options;
  if (request.remote_addrs_size() == 0) {
    return options;
  }

  options.remote_addrs.assign(request.remote_addrs().begin(),
                              request.remote_addrs().end());
  options.src_weight_segments.reserve(request.src_weight_segments_size());
  for (const auto& segment_list : request.src_weight_segments()) {
    std::vector<WeightSegment> segments;
    segments.reserve(segment_list.segments_size());
    for (const auto& segment : segment_list.segments()) {
      segments.emplace_back(segment.offset(), segment.size());
    }
    options.src_weight_segments.emplace_back(std::move(segments));
  }
  return options;
}

}  // namespace xllm::api_service
