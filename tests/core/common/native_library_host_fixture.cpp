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

#include "core/common/native_library.h"

extern "C" __attribute__((visibility("default"))) void
xllm_native_library_fixture_load() {
  xllm::load_native_libraries();
}

extern "C" __attribute__((visibility("default"))) int32_t
xllm_native_library_fixture_host_value() {
  return 42;
}
