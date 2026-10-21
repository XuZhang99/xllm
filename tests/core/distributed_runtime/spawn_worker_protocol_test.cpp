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

#include "core/distributed_runtime/spawn_worker_server/spawn_worker_protocol.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

namespace xllm::spawn_worker_protocol {
namespace {

std::vector<char*> mutable_argv(std::vector<std::string>* arguments) {
  std::vector<char*> argv;
  argv.reserve(arguments->size());
  for (std::string& argument : *arguments) {
    argv.emplace_back(argument.data());
  }
  return argv;
}

TEST(SpawnWorkerProtocolTest, UsesExplicitDtypeFromCurrentArguments) {
  std::vector<std::string> arguments(kArgumentCount, "unused");
  arguments[kIndexerCacheDtypeArgumentIndex] = "int8";
  std::vector<char*> argv = mutable_argv(&arguments);

  const std::optional<std::string> indexer_cache_dtype =
      parse_indexer_cache_dtype(kArgumentCount, argv.data());

  ASSERT_TRUE(indexer_cache_dtype.has_value());
  EXPECT_EQ(indexer_cache_dtype.value(), "int8");
}

TEST(SpawnWorkerProtocolTest, UsesDefaultDtypeFromLegacyArguments) {
  std::vector<std::string> arguments(kMinimumArgumentCount, "unused");
  std::vector<char*> argv = mutable_argv(&arguments);

  const std::optional<std::string> indexer_cache_dtype =
      parse_indexer_cache_dtype(kMinimumArgumentCount, argv.data());

  ASSERT_TRUE(indexer_cache_dtype.has_value());
  EXPECT_EQ(indexer_cache_dtype.value(), kDefaultIndexerCacheDtype);
}

TEST(SpawnWorkerProtocolTest, RejectsArgumentsBelowLegacyMinimum) {
  std::vector<std::string> arguments(kMinimumArgumentCount - 1, "unused");
  std::vector<char*> argv = mutable_argv(&arguments);

  EXPECT_FALSE(parse_indexer_cache_dtype(kMinimumArgumentCount - 1, argv.data())
                   .has_value());
}

TEST(SpawnWorkerProtocolTest, IgnoresUnknownTrailingArguments) {
  std::vector<std::string> arguments(kArgumentCount + 1, "unused");
  arguments[kIndexerCacheDtypeArgumentIndex] = "int8";
  std::vector<char*> argv = mutable_argv(&arguments);

  const std::optional<std::string> indexer_cache_dtype =
      parse_indexer_cache_dtype(kArgumentCount + 1, argv.data());

  ASSERT_TRUE(indexer_cache_dtype.has_value());
  EXPECT_EQ(indexer_cache_dtype.value(), "int8");
}

TEST(SpawnWorkerProtocolTest, PreservesExplicitEmptyDtype) {
  std::vector<std::string> arguments(kArgumentCount, "unused");
  arguments[kIndexerCacheDtypeArgumentIndex] = "";
  std::vector<char*> argv = mutable_argv(&arguments);

  const std::optional<std::string> indexer_cache_dtype =
      parse_indexer_cache_dtype(kArgumentCount, argv.data());

  ASSERT_TRUE(indexer_cache_dtype.has_value());
  EXPECT_TRUE(indexer_cache_dtype->empty());
}

TEST(SpawnWorkerProtocolTest, PreservesExplicitModelImplementation) {
  std::vector<std::string> arguments(kArgumentCount, "unused");
  arguments[kModelImplArgumentIndex] = "python";
  std::vector<char*> argv = mutable_argv(&arguments);

  EXPECT_EQ(parse_model_impl(kArgumentCount, argv.data()), "python");
}

TEST(SpawnWorkerProtocolTest, PreservesExplicitEmptyModelImplementation) {
  std::vector<std::string> arguments(kArgumentCount, "unused");
  arguments[kModelImplArgumentIndex] = "";
  std::vector<char*> argv = mutable_argv(&arguments);

  EXPECT_TRUE(parse_model_impl(kArgumentCount, argv.data()).empty());
}

TEST(SpawnWorkerProtocolTest, DefaultsModelImplementationForLegacyArguments) {
  for (int32_t argc = kMinimumArgumentCount; argc <= kModelImplArgumentIndex;
       ++argc) {
    std::vector<std::string> arguments(argc, "unused");
    std::vector<char*> argv = mutable_argv(&arguments);

    EXPECT_TRUE(parse_model_impl(argc, argv.data()).empty());
  }
}

TEST(SpawnWorkerProtocolTest,
     PreservesModelImplementationWithTrailingArguments) {
  std::vector<std::string> arguments(kArgumentCount + 1, "unused");
  arguments[kModelImplArgumentIndex] = "python";
  std::vector<char*> argv = mutable_argv(&arguments);

  EXPECT_EQ(parse_model_impl(kArgumentCount + 1, argv.data()), "python");
}

TEST(SpawnWorkerProtocolTest, PreservesIndependentExtensionImplementation) {
  std::vector<std::string> arguments(kArgumentCount, "unused");
  arguments[kModelImplArgumentIndex] = "python";
  arguments[kExtensionImplArgumentIndex] = "custom";
  std::vector<char*> argv = mutable_argv(&arguments);

  EXPECT_EQ(parse_model_impl(kArgumentCount, argv.data()), "python");
  EXPECT_EQ(parse_extension_impl(kArgumentCount, argv.data()), "custom");
}

TEST(SpawnWorkerProtocolTest, PreservesExplicitEmptyExtensionImplementation) {
  std::vector<std::string> arguments(kArgumentCount, "unused");
  arguments[kExtensionImplArgumentIndex] = "";
  std::vector<char*> argv = mutable_argv(&arguments);

  EXPECT_TRUE(parse_extension_impl(kArgumentCount, argv.data()).empty());
}

TEST(SpawnWorkerProtocolTest,
     DefaultsExtensionImplementationForLegacyArguments) {
  for (int32_t argc = kMinimumArgumentCount;
       argc <= kExtensionImplArgumentIndex;
       ++argc) {
    std::vector<std::string> arguments(argc, "unused");
    std::vector<char*> argv = mutable_argv(&arguments);

    EXPECT_TRUE(parse_extension_impl(argc, argv.data()).empty());
  }
}

TEST(SpawnWorkerProtocolTest, DefaultsExtensionImplementationForNullArguments) {
  std::vector<std::string> arguments(kArgumentCount, "unused");
  std::vector<char*> argv = mutable_argv(&arguments);
  argv[kExtensionImplArgumentIndex] = nullptr;

  EXPECT_TRUE(parse_extension_impl(kArgumentCount, nullptr).empty());
  EXPECT_TRUE(parse_extension_impl(kArgumentCount, argv.data()).empty());
}

TEST(SpawnWorkerProtocolTest,
     PreservesExtensionImplementationWithTrailingArguments) {
  std::vector<std::string> arguments(kArgumentCount + 1, "unused");
  arguments[kExtensionImplArgumentIndex] = "custom";
  std::vector<char*> argv = mutable_argv(&arguments);

  EXPECT_EQ(parse_extension_impl(kArgumentCount + 1, argv.data()), "custom");
}

}  // namespace
}  // namespace xllm::spawn_worker_protocol
