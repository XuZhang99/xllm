/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <filesystem>
#include <optional>
#include <string>

namespace xllm {

class JsonReader;

// Resolves the model type of a checkpoint from an already parsed config.json.
// The `backend` argument disambiguates checkpoints that inherit their model
// type (see the qwen3_5 handling below).
std::string get_model_type(const JsonReader& reader,
                           const std::filesystem::path& model_path,
                           std::optional<std::string> backend = std::nullopt);

// Resolves the model type of a checkpoint from its config.json, falling back to
// model_index.json and the DiT layout discovery for non-HF checkpoints.
std::string get_model_type(const std::filesystem::path& model_path,
                           std::optional<std::string> backend = std::nullopt);

// Resolves the backend name ("llm", "vlm", "dit", ...) of a checkpoint. DiT
// checkpoints are recognized from their model_index.json / component layout,
// every other checkpoint is mapped through ModelRegistry from its model type.
std::string get_model_backend(const std::filesystem::path& model_path);

// Whether MLA attention is enabled for the checkpoint. DiT backends never use
// MLA. When `backend` is empty it is resolved through get_model_backend.
bool should_enable_mla(
    const std::filesystem::path& model_path,
    const std::optional<std::string>& backend = std::nullopt);

}  // namespace xllm
