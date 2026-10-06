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
#include <string>

#include "xllm_service.pb.h"

namespace xllm {

class MasterManager;

// Adapts control RPC and HTTP requests to the shared master manager.
// MasterManager owns lifecycle transitions and request admission state.
class ControlServiceImpl final {
 public:
  explicit ControlServiceImpl(std::shared_ptr<MasterManager> master_manager);

  void fork_master(::google::protobuf::RpcController* controller,
                   const proto::MasterInfos* request,
                   proto::Status* response,
                   ::google::protobuf::Closure* done);

  void fork_master_http(::google::protobuf::RpcController* controller,
                        const proto::HttpRequest* request,
                        proto::HttpResponse* response,
                        ::google::protobuf::Closure* done);

  void sleep(::google::protobuf::RpcController* controller,
             const proto::MasterInfos* request,
             proto::Status* response,
             ::google::protobuf::Closure* done);

  void sleep_http(::google::protobuf::RpcController* controller,
                  const proto::HttpRequest* request,
                  proto::HttpResponse* response,
                  ::google::protobuf::Closure* done);

  void wakeup(::google::protobuf::RpcController* controller,
              const proto::MasterInfos* request,
              proto::Status* response,
              ::google::protobuf::Closure* done);

  void wakeup_http(::google::protobuf::RpcController* controller,
                   const proto::HttpRequest* request,
                   proto::HttpResponse* response,
                   ::google::protobuf::Closure* done);

  void start_profile_http(::google::protobuf::RpcController* controller,
                          const proto::HttpRequest* request,
                          proto::HttpResponse* response,
                          ::google::protobuf::Closure* done);

  void stop_profile_http(::google::protobuf::RpcController* controller,
                         const proto::HttpRequest* request,
                         proto::HttpResponse* response,
                         ::google::protobuf::Closure* done);

  void link_p2p(::google::protobuf::RpcController* controller,
                const proto::P2PLinkRequest* request,
                proto::Status* response,
                ::google::protobuf::Closure* done);

  void link_p2p_http(::google::protobuf::RpcController* controller,
                     const proto::HttpRequest* request,
                     proto::HttpResponse* response,
                     ::google::protobuf::Closure* done);

  void unlink_p2p(::google::protobuf::RpcController* controller,
                  const proto::P2PLinkRequest* request,
                  proto::Status* response,
                  ::google::protobuf::Closure* done);

  void unlink_p2p_http(::google::protobuf::RpcController* controller,
                       const proto::HttpRequest* request,
                       proto::HttpResponse* response,
                       ::google::protobuf::Closure* done);

 private:
  bool do_fork_master(const proto::MasterInfos& request,
                      std::string* error_message);
  bool do_sleep(const proto::MasterInfos& request, std::string* error_message);
  bool do_wakeup(const proto::MasterInfos& request, std::string* error_message);

  std::shared_ptr<MasterManager> master_manager_;
};

}  // namespace xllm
