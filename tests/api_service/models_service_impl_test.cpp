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

#include "xllm/api_service/models_service_impl.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "api_service/openai_json.h"

namespace xllm {
namespace {

TEST(ModelsServiceImplTest, OpenAIModelCardIncludesLoadedConfiguration) {
  ModelsServiceImpl service(
      {"alias"}, {"repository"}, {"1"}, "/models/weights", 8192);
  proto::ModelListResponse response;
  ASSERT_TRUE(service.list_models(nullptr, &response));
  const auto json = api_service::openai_models_json(response);
  const auto& card = json["data"][0];
  EXPECT_EQ(card["id"], "alias");
  EXPECT_EQ(card["owned_by"], "xllm");
  EXPECT_EQ(card["root"], "/models/weights");
  EXPECT_EQ(card["max_model_len"], 8192);
  EXPECT_TRUE(card["parent"].is_null());
  ASSERT_EQ(card["permission"].size(), 1);
  const auto& permission = card["permission"][0];
  EXPECT_EQ(permission["object"], "model_permission");
  EXPECT_TRUE(permission["id"].get<std::string>().starts_with("modelperm-"));
  EXPECT_EQ(permission["created"], card["created"]);
  EXPECT_EQ(permission["organization"], "*");
  EXPECT_TRUE(permission["group"].is_null());
  for (const char* key : {"allow_sampling", "allow_logprobs", "allow_view"}) {
    EXPECT_EQ(permission[key], true);
  }
  for (const char* key : {"allow_create_engine",
                          "allow_search_indices",
                          "allow_fine_tuning",
                          "is_blocking"}) {
    EXPECT_EQ(permission[key], false);
  }
}

TEST(ModelsServiceImplTest, UnknownModelMetadataIsNull) {
  ModelsServiceImpl service({"alias"}, {"repository"}, {"1"});
  proto::ModelListResponse response;
  ASSERT_TRUE(service.list_models(nullptr, &response));
  const auto card = api_service::openai_models_json(response)["data"][0];
  EXPECT_TRUE(card["root"].is_null());
  EXPECT_TRUE(card["max_model_len"].is_null());
}

TEST(ModelsServiceImplTest, RepositoryIndexUsesRepositoryMetadata) {
  const std::vector<std::string> model_names = {"GLM-5.1", "Qwen3-8B"};
  const std::vector<std::string> model_repository_names = {"glm-51-w8a8-npu",
                                                           "qwen3"};
  const std::vector<std::string> model_versions = {"2", "3"};
  ModelsServiceImpl service(
      model_names, model_repository_names, model_versions);

  proto::ModelListRequest request;
  proto::ModelListResponse response;
  ASSERT_TRUE(service.list_models(&request, &response));
  EXPECT_EQ(response.object(), "list");
  ASSERT_EQ(response.data_size(), static_cast<int>(model_names.size()));

  const nlohmann::json repository_index =
      nlohmann::json::parse(service.list_model_versions());
  ASSERT_TRUE(repository_index.is_array());
  ASSERT_EQ(repository_index.size(), model_names.size());

  for (std::size_t i = 0; i < model_names.size(); ++i) {
    ASSERT_EQ(response.data(static_cast<int>(i)).id(), model_names[i]);
    ASSERT_TRUE(repository_index[i].is_object());
    ASSERT_TRUE(repository_index[i].contains("name"));
    ASSERT_TRUE(repository_index[i].contains("version"));
    ASSERT_TRUE(repository_index[i].contains("state"));
    ASSERT_TRUE(repository_index[i].contains("reason"));
    EXPECT_EQ(repository_index[i]["name"], model_repository_names[i]);
    EXPECT_EQ(repository_index[i]["version"], model_versions[i]);
    EXPECT_EQ(repository_index[i]["state"], "READY");
    EXPECT_EQ(repository_index[i]["reason"], "normal");
  }
}

}  // namespace
}  // namespace xllm
