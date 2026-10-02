/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstdint>
#include <optional>
#include <string>

#include "common/options.h"
#include "common/types.h"

namespace xllm {

// Validate startup combinations of parallel layout, runtime configuration and
// registered model capabilities before distributed workers are initialized.
std::optional<std::string> validate_context_parallel_config(
    const Options& options,
    EngineType engine_type,
    const std::string& model_type,
    int32_t global_world_size);

void validate_layerwise_split_size_startup_config(const Options& options,
                                                  const std::string& model_type,
                                                  int32_t global_world_size);

}  // namespace xllm
