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

#include <string>
#include <unordered_map>
#include <unordered_set>

namespace xllm {

// Records the component registries supplied by each extension family. Like the
// factory registries, registrations finish during startup before concurrent
// use.
class ExtensionRegistry final {
 public:
  static ExtensionRegistry& get_instance();

  ExtensionRegistry(const ExtensionRegistry&) = delete;
  ExtensionRegistry& operator=(const ExtensionRegistry&) = delete;

  void register_implementation(std::string implementation,
                               std::string component);

  bool has_implementation(const std::string& implementation) const;
  bool has_implementation(const std::string& implementation,
                          const std::string& component) const;

 private:
  ExtensionRegistry() = default;

  std::unordered_map<std::string, std::unordered_set<std::string>> components_;
};

}  // namespace xllm
