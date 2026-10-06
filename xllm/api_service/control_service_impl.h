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

#pragma once

#include <memory>

#include "core/common/types.h"
#include "xllm_service.pb.h"

namespace xllm {

class MasterManager;
class Options;

// Processes typed control requests using the shared master manager.
// MasterManager owns lifecycle transitions and request admission state.
class ControlServiceImpl final {
 public:
  explicit ControlServiceImpl(std::shared_ptr<MasterManager> master_manager);

  Status fork_master(const proto::MasterInfos& request);
  Status sleep(const proto::MasterInfos& request);
  Status wakeup(const proto::MasterInfos& request);
  Status start_profile();
  Status stop_profile();
  Status link_p2p(const proto::P2PLinkRequest& request);
  Status unlink_p2p(const proto::P2PLinkRequest& request);

 private:
  friend class ControlServiceImplTest;

  static Status parse_fork_master_request(const proto::MasterInfos& request,
                                          Options& options);
  static WakeupOptions parse_wakeup_options(const proto::MasterInfos& request);

  std::shared_ptr<MasterManager> master_manager_;
};

}  // namespace xllm
