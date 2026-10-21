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

#include <dlfcn.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>

namespace xllm {
namespace {

TEST(NativeLibraryLocalHostTest, PromotesLocalHostBeforeLoadingPlugin) {
  EXPECT_EXIT(
      {
        setenv("XLLM_NATIVE_LIBRARIES", NATIVE_HOST_DEPENDENT_FIXTURE, 1);
        void* host = dlopen(NATIVE_HOST_FIXTURE, RTLD_NOW | RTLD_LOCAL);
        if (host == nullptr ||
            dlsym(RTLD_DEFAULT, "xllm_native_library_fixture_host_value") !=
                nullptr) {
          std::exit(1);
        }
        using Loader = void (*)();
        auto load = reinterpret_cast<Loader>(
            dlsym(host, "xllm_native_library_fixture_load"));
        if (load == nullptr) {
          std::exit(2);
        }
        load();
        if (dlsym(RTLD_DEFAULT, "xllm_native_library_fixture_host_value") ==
            nullptr) {
          std::exit(3);
        }
        void* plugin =
            dlopen(NATIVE_HOST_DEPENDENT_FIXTURE, RTLD_NOW | RTLD_NOLOAD);
        if (plugin == nullptr) {
          std::exit(4);
        }
        using ValueGetter = int32_t (*)();
        auto getter = reinterpret_cast<ValueGetter>(
            dlsym(plugin, "xllm_native_library_fixture_value"));
        const bool valid = getter != nullptr && getter() == 42;
        dlclose(plugin);
        dlclose(host);
        std::exit(valid ? 0 : 5);
      },
      testing::ExitedWithCode(0),
      "");
}

}  // namespace
}  // namespace xllm
