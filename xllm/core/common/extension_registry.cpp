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

#include "core/common/extension_registry.h"

#include <utility>

namespace xllm {

ExtensionRegistry& ExtensionRegistry::get_instance() {
  static ExtensionRegistry instance;
  return instance;
}

void ExtensionRegistry::register_implementation(std::string implementation,
                                                std::string component) {
  components_[std::move(implementation)].insert(std::move(component));
}

bool ExtensionRegistry::has_implementation(
    const std::string& implementation) const {
  return components_.find(implementation) != components_.end();
}

bool ExtensionRegistry::has_implementation(const std::string& implementation,
                                           const std::string& component) const {
  const auto it = components_.find(implementation);
  return it != components_.end() && it->second.count(component) != 0;
}

}  // namespace xllm
