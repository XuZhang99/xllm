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

#include <brpc/channel.h>
#include <brpc/server.h>
#include <gtest/gtest.h>

#include "api_service/non_stream_call.h"
#include "api_service/openai_batch.h"
#include "api_service/openai_json.h"
#include "api_service/openai_request.h"
#include "api_service/stream_call.h"
#include "xllm_service.pb.h"

namespace xllm::api_service {
namespace {

TEST(OpenAIRequestTest, ChatAliasesAndNullableDefaults) {
  auto [status, body] = normalize_openai_request(
      R"({"messages":[{"role":"assistant","content":null,"reasoning":"why"}],
          "max_tokens":99,"max_completion_tokens":7,"stop":"END",
          "temperature":null,"tool_choice":null})",
      OpenAIEndpoint::CHAT,
      "model");
  ASSERT_TRUE(status.ok()) << status.message();
  const auto json = nlohmann::json::parse(body);
  EXPECT_EQ(json["max_tokens"], 7);
  EXPECT_EQ(json["stop"], nlohmann::json::array({"END"}));
  EXPECT_EQ(json["messages"][0]["reasoning_content"], "why");
  EXPECT_EQ(json["temperature"], 1.0);
  EXPECT_EQ(json["model"], "model");
  EXPECT_FALSE(json.contains("tool_choice"));
}

TEST(OpenAIRequestTest, BatchedInputsKeepTheirTypesAndOrder) {
  for (const auto& [endpoint, field, target] :
       {std::tuple{OpenAIEndpoint::COMPLETION, "prompt", "prompts"},
        std::tuple{OpenAIEndpoint::EMBEDDING, "input", "inputs"}}) {
    for (const auto& input : {nlohmann::json::array({"a", "b"}),
                              nlohmann::json::array({{1, 2}, {3}})}) {
      auto [status, body] = normalize_openai_request(
          nlohmann::json({{field, input}}).dump(), endpoint, "model");
      ASSERT_TRUE(status.ok()) << status.message();
      const auto json = nlohmann::json::parse(body);
      ASSERT_EQ(json[target].size(), 2);
      EXPECT_EQ(json[target][0][input[0].is_string() ? "text" : "token_ids"],
                input[0]);
      EXPECT_EQ(json[target][1][input[1].is_string() ? "text" : "token_ids"],
                input[1]);
    }
  }
}

TEST(OpenAIRequestTest, InvalidParametersAreClientErrors) {
  const auto base =
      nlohmann::json::parse(R"({"messages":[{"role":"user","content":"hi"}]})");
  for (const auto& patch :
       {nlohmann::json{{"stop", 3}},
        {{"stop", ""}},
        {{"top_p", 0}},
        {{"n", -1}},
        {{"max_completion_tokens", 0}},
        {{"max_completion_tokens", 1.5}},
        {{"stream", "true"}},
        {{"stream_options", {{"include_usage", true}}}},
        {{"stream", true}, {"stream_options", {{"include_usage", 1}}}},
        {{"top_logprobs", 2}, {"logprobs", false}},
        {{"tools", nlohmann::json::array()}},
        {{"tool_choice", "auto"}},
        {{"messages", nlohmann::json::array()}}}) {
    auto request = base;
    request.update(patch);
    auto [status, body] =
        normalize_openai_request(request.dump(), OpenAIEndpoint::CHAT, "model");
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT) << request;
  }
}

TEST(OpenAIRequestTest, GreedySamplingRequiresOneChoice) {
  for (const OpenAIEndpoint endpoint :
       {OpenAIEndpoint::CHAT, OpenAIEndpoint::COMPLETION}) {
    for (const bool stream : {false, true}) {
      for (const uint32_t n : {1U, 2U}) {
        for (const auto& temperature : {nlohmann::json(0),
                                        nlohmann::json(0.5),
                                        nlohmann::json(1e-8),
                                        nlohmann::json(nullptr)}) {
          nlohmann::json request = {
              {"prompt", {"hello", "world"}},
              {"messages", {{{"role", "user"}, {"content", "hello"}}}},
              {"temperature", temperature},
              {"n", n},
              {"stream", stream}};
          std::string param = "stale";
          const auto [status, body] = normalize_openai_request(
              request.dump(), endpoint, "model", &param);
          const bool greedy_multiple = temperature == 0 && n > 1;
          EXPECT_EQ(status.ok(), !greedy_multiple) << request;
          EXPECT_TRUE(param.empty());
          if (status.ok() && temperature == 1e-8) {
            EXPECT_EQ(nlohmann::json::parse(body)["temperature"], 0.01);
          }
        }
      }
    }
  }
}

TEST(OpenAIRequestTest, Vllm023AcceptsTemperaturesAboveTwo) {
  for (const OpenAIEndpoint endpoint :
       {OpenAIEndpoint::CHAT, OpenAIEndpoint::COMPLETION}) {
    const auto [status, body] = normalize_openai_request(
        R"({"prompt":"hi","messages":[{"role":"user","content":"hi"}],"temperature":3})",
        endpoint,
        "model");
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_EQ(nlohmann::json::parse(body)["temperature"], 3);
  }
}

TEST(OpenAIRequestTest, SchemaAndSamplingErrorsHaveDistinctTypes) {
  for (const auto& [body, expected_type] :
       {std::pair{"{", "Bad Request"},
        {R"({"prompt":[1,"hi"]})", "Bad Request"},
        {R"({"prompt":[]})", "BadRequestError"},
        {R"({"prompt":"hi","stream_options":{"include_usage":true}})",
         "Bad Request"},
        {R"({"prompt":"hi","temperature":-1})", "BadRequestError"},
        {R"({"prompt":"hi","temperature":0,"n":2})", "BadRequestError"}}) {
    bool schema_error = false;
    std::string param;
    const auto [status, normalized] = normalize_openai_request(
        body, OpenAIEndpoint::COMPLETION, "model", &param, &schema_error);
    ASSERT_FALSE(status.ok());
    const auto error = nlohmann::json::parse(openai_error_json(
        status.code(), status.message(), param, schema_error));
    EXPECT_EQ(error["error"]["type"], expected_type);
  }
  bool schema_error = true;
  const auto [status, body] = normalize_openai_request(R"({"messages":[]})",
                                                       OpenAIEndpoint::CHAT,
                                                       "model",
                                                       nullptr,
                                                       &schema_error);
  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(schema_error);
}

TEST(OpenAIRequestTest, ValidationErrorsIdentifyTheirParameters) {
  const nlohmann::json base = {
      {"messages", {{{"role", "user"}, {"content", "hello"}}}}};
  for (const auto& [patch, expected] :
       std::vector<std::pair<nlohmann::json, std::string>>{
           {{{"max_tokens", -1}}, "max_tokens"},
           {{{"max_tokens", 8}, {"max_completion_tokens", -1}}, "max_tokens"},
           {{{"temperature", -1}}, "temperature"},
           {{{"top_p", 0}}, "top_p"},
           {{{"stream_options", {{"include_usage", true}}}}, "stream_options"},
           {{{"top_logprobs", 2}}, "top_logprobs"},
           {{{"tool_choice", "auto"}}, "tool_choice"},
           {{{"n", 0}}, ""}}) {
    auto request = base;
    request.update(patch);
    std::string param = "stale";
    const auto [status, body] = normalize_openai_request(
        request.dump(), OpenAIEndpoint::CHAT, "model", &param);
    ASSERT_FALSE(status.ok()) << request;
    EXPECT_EQ(param, expected) << request;
    const auto error = nlohmann::json::parse(
        openai_error_json(status.code(), status.message(), param));
    EXPECT_EQ(
        error["error"]["param"],
        expected.empty() ? nlohmann::json(nullptr) : nlohmann::json(expected));
  }
  for (const auto& [choice, expected] :
       std::vector<std::pair<nlohmann::json, std::string>>{
           {42, "tool_choice"},
           {{{"type", "function"}}, "tool_choice.function"},
           {{{"type", "function"}, {"function", nlohmann::json::object()}},
            "tool_choice.function.name"},
           {{{"type", "function"}, {"function", {{"name", "missing"}}}},
            "tool_choice"}}) {
    auto request = base;
    request["tools"] = {
        {{"type", "function"}, {"function", {{"name", "weather"}}}}};
    request["tool_choice"] = choice;
    std::string param;
    EXPECT_FALSE(normalize_openai_request(
                     request.dump(), OpenAIEndpoint::CHAT, "model", &param)
                     .first.ok());
    EXPECT_EQ(param, expected);
  }
  std::string param = "stale";
  EXPECT_FALSE(
      normalize_openai_request("{", OpenAIEndpoint::CHAT, "model", &param)
          .first.ok());
  EXPECT_TRUE(param.empty());
  const auto [status, body] = normalize_openai_request(
      R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":-1,"max_completion_tokens":8})",
      OpenAIEndpoint::CHAT,
      "model",
      &param);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(nlohmann::json::parse(body)["max_tokens"], 8);
}

TEST(OpenAIResponseTest, NamedToolChoicePreservesStopAndLength) {
  for (const bool stream : {false, true}) {
    for (const std::string reason : {"tool_calls", "length", "stop"}) {
      proto::ChatResponse response;
      response.set_object(stream ? "chat.completion.chunk" : "chat.completion");
      response.add_choices()->set_finish_reason(reason);
      const auto named = openai_response_json(response, true);
      EXPECT_EQ(named["choices"][0]["finish_reason"],
                reason == "tool_calls" ? "stop" : reason);
      const auto automatic = openai_response_json(response);
      EXPECT_EQ(automatic["choices"][0]["finish_reason"], reason);
    }
  }
}

TEST(OpenAIRequestTest, CompletionDefaultsAndExtendedStops) {
  auto [status, body] = normalize_openai_request(
      R"({"prompt":"hi","max_tokens":null,"stop":["1","2","3","4","5"],"frequency_penalty":-1})",
      OpenAIEndpoint::COMPLETION,
      "model");
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(nlohmann::json::parse(body)["max_tokens"], 16);
}

TEST(OpenAIRequestTest, MalformedPromptBatchesAreRejected) {
  for (const auto& input : {R"([])",
                            R"([1,"x"])",
                            R"([[-1]])",
                            R"([[2147483648]])",
                            R"(["a",[1]])"}) {
    auto [status, body] =
        normalize_openai_request(std::string("{\"input\":") + input + "}",
                                 OpenAIEndpoint::EMBEDDING,
                                 "model");
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT) << input;
  }
}

TEST(OpenAIRequestTest, UnsupportedSamplingControlsFailExplicitly) {
  for (const char* field :
       {"seed", "min_p", "min_tokens", "logit_bias", "structured_outputs"}) {
    auto request = nlohmann::json({{"prompt", "hi"}, {field, 1}});
    const auto [status, body] = normalize_openai_request(
        request.dump(), OpenAIEndpoint::COMPLETION, "model");
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT) << field;
    EXPECT_NE(status.message().find(field), std::string::npos);
  }
}

TEST(OpenAIRequestTest, InternalInputsAndOversizedBatchesAreRejected) {
  for (const auto& request :
       {nlohmann::json{{"prompt", "hi"}, {"prompts", {{{"text", "hidden"}}}}},
        nlohmann::json{{"prompt", std::vector<std::string>(1025, "hi")}}}) {
    EXPECT_EQ(normalize_openai_request(
                  request.dump(), OpenAIEndpoint::COMPLETION, "model")
                  .first.code(),
              StatusCode::INVALID_ARGUMENT);
  }
  EXPECT_EQ(
      normalize_openai_request(
          R"({"input":"","token_ids":[1]})", OpenAIEndpoint::EMBEDDING, "model")
          .first.code(),
      StatusCode::INVALID_ARGUMENT);
}

TEST(OpenAIRequestTest, CompletionTokenInputPreservesExistingExtension) {
  auto [status, body] = normalize_openai_request(
      R"({"prompt":[1,2,3]})", OpenAIEndpoint::COMPLETION, "model");
  ASSERT_TRUE(status.ok()) << status.message();
  const auto json = nlohmann::json::parse(body);
  EXPECT_EQ(json["token_ids"], nlohmann::json::array({1, 2, 3}));
  EXPECT_FALSE(json.contains("prompts"));
  std::tie(status, body) =
      normalize_openai_request(R"({"prompt":"","token_ids":[1,2,3]})",
                               OpenAIEndpoint::COMPLETION,
                               "model");
  EXPECT_TRUE(status.ok()) << status.message();
}

TEST(OpenAIRequestTest,
     InferenceBodyLengthIsCheckedWithoutRequiringContentLength) {
  brpc::Controller controller;
  controller.request_attachment().append("{}binary");
  controller.http_request().SetHeader(kInferContentLength, "2");
  auto [status, body] = openai_request_body(controller);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(body, "{}");
  for (const char* value : {"-1", "9999", "nan", "2x"}) {
    controller.http_request().SetHeader(kInferContentLength, value);
    std::tie(status, body) = openai_request_body(controller);
    EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
    std::tie(status, body) =
        openai_request_body(controller, /*binary_input=*/false);
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(body, "{}binary");
  }
}

TEST(OpenAIRequestTest, CallPayloadParsingHandlesInvalidLengthHeaders) {
  for (const char* value :
       {"-1", "9999", "nan", "2x", "184467440737095516160"}) {
    for (const char* header : {kInferContentLength, kContentLength}) {
      brpc::Controller controller;
      controller.request_attachment().append("{}binary");
      controller.http_request().SetHeader(kInferContentLength, "2");
      controller.http_request().SetHeader(kContentLength, "8");
      controller.http_request().SetHeader(header, value);
      StreamCall<proto::ChatRequest, proto::ChatResponse> call(
          &controller,
          brpc::DoNothing(),
          new proto::ChatRequest(),
          new proto::ChatResponse(),
          /*use_arena=*/false,
          /*is_http_request=*/true);
      EXPECT_TRUE(call.take_request_payload().empty());
    }
  }
  brpc::Controller controller;
  controller.request_attachment().append("{}binary");
  controller.http_request().SetHeader(kInferContentLength, "2");
  controller.http_request().SetHeader(kContentLength, "8");
  StreamCall<proto::ChatRequest, proto::ChatResponse> call(
      &controller,
      brpc::DoNothing(),
      new proto::ChatRequest(),
      new proto::ChatResponse(),
      /*use_arena=*/false,
      /*is_http_request=*/true);
  EXPECT_EQ(call.take_request_payload(), "binary");
}

TEST(OpenAIJsonTest, ChatFinishAndUsageChunksKeepRequiredFields) {
  proto::ChatResponse response;
  response.set_object("chat.completion.chunk");
  auto* choice = response.add_choices();
  choice->set_index(0);
  choice->set_finish_reason("stop");
  auto json = openai_response_json(response);
  EXPECT_TRUE(json["choices"][0]["delta"].is_object());
  EXPECT_TRUE(json["choices"][0]["delta"].empty());
  EXPECT_TRUE(json["choices"][0]["logprobs"].is_null());
  response.clear_choices();
  response.mutable_usage()->set_prompt_tokens(2);
  json = openai_response_json(response);
  EXPECT_EQ(json["choices"], nlohmann::json::array());
  EXPECT_EQ(json["usage"]["prompt_tokens"], 2);
}

TEST(OpenAIJsonTest, ReasoningAndLogprobBytesMatchVllm) {
  proto::ChatResponse response;
  auto* choice = response.add_choices();
  choice->mutable_message()->set_reasoning_content("why");
  auto* logprob = choice->mutable_logprobs()->add_content();
  logprob->set_token("你");
  logprob->set_logprob(-0.5);
  const auto json = openai_response_json(response);
  EXPECT_EQ(json["choices"][0]["message"]["reasoning"], "why");
  EXPECT_FALSE(json["choices"][0]["message"].contains("reasoning_content"));
  EXPECT_FALSE(
      json["choices"][0]["logprobs"]["content"][0].contains("token_id"));
  EXPECT_EQ(json["choices"][0]["logprobs"]["content"][0]["bytes"],
            nlohmann::json::array({228, 189, 160}));
  EXPECT_EQ(json["choices"][0]["logprobs"]["content"][0]["top_logprobs"],
            nlohmann::json::array());
}

TEST(OpenAIJsonTest, FullChatPreservesNullsAndForcedToolContent) {
  proto::ChatResponse response;
  response.set_object("chat.completion");
  auto* message = response.add_choices()->mutable_message();
  message->set_role("assistant");
  message->set_content("");
  message->set_reasoning_content("thinking");
  response.mutable_usage()->mutable_prompt_tokens_details()->set_cached_tokens(
      0);
  auto json = openai_response_json(response);
  EXPECT_TRUE(json["choices"][0]["message"]["content"].is_null());
  EXPECT_FALSE(json["choices"][0]["message"].contains("tool_calls"));
  EXPECT_TRUE(json["choices"][0]["message"].contains("refusal"));
  EXPECT_TRUE(json.contains("system_fingerprint"));
  EXPECT_TRUE(json["usage"]["prompt_tokens_details"].is_null());
  EXPECT_TRUE(json["usage"].contains("completion_tokens_details"));
  message->add_tool_calls()->mutable_function()->set_name("weather");
  for (const bool named : {false, true}) {
    json = openai_response_json(response, named, !named);
    EXPECT_EQ(json["choices"][0]["message"]["content"], "");
  }
  response.set_object("chat.completion.chunk");
  response.mutable_choices(0)->mutable_delta()->set_reasoning_content("");
  json = openai_response_json(response);
  EXPECT_FALSE(json.contains("service_tier"));
  EXPECT_TRUE(json["choices"][0]["delta"].empty());
  EXPECT_FALSE(json["usage"].contains("prompt_tokens_details"));
  EXPECT_FALSE(json["usage"].contains("completion_tokens_details"));
}

TEST(OpenAIJsonTest, CompletionSeparatesFullAndStreamMetadata) {
  proto::CompletionResponse response;
  response.set_object("text_completion");
  response.add_choices()->mutable_logprobs()->add_token_ids(42);
  response.mutable_usage()->set_prompt_tokens(2);
  for (const bool stream : {false, true}) {
    const auto json = openai_response_json(response, stream);
    EXPECT_EQ(json.contains("system_fingerprint"), !stream);
    EXPECT_EQ(json["choices"][0].contains("prompt_logprobs"), !stream);
    EXPECT_EQ(json["usage"].contains("prompt_tokens_details"), !stream);
    EXPECT_FALSE(json["choices"][0]["logprobs"].contains("token_ids"));
  }
}

TEST(OpenAIJsonTest, ToolArgumentDeltasOnlyRepeatIndexAndArguments) {
  proto::ChatResponse response;
  response.set_object("chat.completion.chunk");
  auto* delta = response.add_choices()->mutable_delta();
  auto* tool = delta->add_tool_calls();
  tool->set_index(0);
  tool->set_id("call_test");
  tool->mutable_function()->set_name("weather");
  auto json = openai_response_json(response);
  EXPECT_TRUE(json["choices"][0]["delta"]["content"].is_null());
  EXPECT_EQ(json["choices"][0]["delta"]["tool_calls"][0]["type"], "function");
  tool->clear_id();
  tool->mutable_function()->clear_name();
  tool->mutable_function()->set_arguments("{}");
  json = openai_response_json(response);
  const auto& call = json["choices"][0]["delta"]["tool_calls"][0];
  EXPECT_EQ(
      call,
      (nlohmann::json{{"index", 0}, {"function", {{"arguments", "{}"}}}}));
}

TEST(OpenAIJsonTest, EmbeddingsUseFloat32LittleEndianBase64) {
  proto::EmbeddingResponse response;
  response.add_data()->add_embedding(1.0f);
  response.mutable_data(0)->add_embedding(-2.0f);
  const auto json = openai_embedding_json(response, "base64");
  EXPECT_EQ(json["data"][0]["embedding"], "AACAPwAAAMA=");
  EXPECT_FALSE(json["usage"].contains("completion_tokens"));
  EXPECT_EQ(openai_embedding_json(response, "float")["data"][0]["embedding"],
            nlohmann::json::array({1.0, -2.0}));
}

TEST(OpenAIBatchTest, ReordersChoicesAndSumsFinalUsage) {
  OpenAIBatch batch(/*size=*/2, /*choices_per_prompt=*/2, /*streaming=*/false);
  std::vector<RequestOutput> sent;
  OutputCallback send = [&sent](RequestOutput output) {
    sent.emplace_back(std::move(output));
    return true;
  };
  for (size_t index : {1, 0}) {
    RequestOutput output;
    output.finished = true;
    output.usage = Usage{3, 2, 5, 1};
    output.outputs.resize(2);
    output.outputs[0].index = 0;
    output.outputs[1].index = 1;
    batch.accept(index, std::move(output), send);
  }
  ASSERT_EQ(sent.size(), 1);
  ASSERT_EQ(sent[0].outputs.size(), 4);
  EXPECT_EQ(sent[0].outputs[0].index, 0);
  EXPECT_EQ(sent[0].outputs[3].index, 3);
  EXPECT_EQ(sent[0].usage->num_prompt_tokens, 6);
  EXPECT_EQ(sent[0].usage->num_total_tokens, 10);
  EXPECT_TRUE(sent[0].finished);
}

TEST(OpenAIBatchTest, StreamingWaitsForEveryPromptAndUsesLatestUsage) {
  OpenAIBatch batch(/*size=*/2, /*choices_per_prompt=*/1, /*streaming=*/true);
  std::vector<RequestOutput> sent;
  OutputCallback send = [&sent](RequestOutput output) {
    sent.emplace_back(std::move(output));
    return true;
  };
  RequestOutput first;
  first.usage = Usage{3, 1, 4, 0};
  batch.accept(0, std::move(first), send);
  RequestOutput second;
  second.finished = true;
  second.usage = Usage{4, 2, 6, 0};
  batch.accept(1, std::move(second), send);
  RequestOutput last;
  last.finished = true;
  last.usage = Usage{3, 3, 6, 0};
  batch.accept(0, std::move(last), send);
  ASSERT_EQ(sent.size(), 3);
  EXPECT_FALSE(sent[0].finished);
  EXPECT_FALSE(sent[1].finished);
  EXPECT_TRUE(sent[2].finished);
  EXPECT_EQ(sent[2].usage->num_total_tokens, 12);
}

TEST(OpenAIBatchTest, ErrorClosesAllRemainingCallbacks) {
  OpenAIBatch batch(/*size=*/2, /*choices_per_prompt=*/1, /*streaming=*/true);
  int32_t sent = 0;
  OutputCallback send = [&sent](RequestOutput) {
    ++sent;
    return true;
  };
  EXPECT_FALSE(batch.accept(
      0, RequestOutput(Status(StatusCode::INVALID_ARGUMENT, "bad")), send));
  EXPECT_FALSE(batch.accept(1, RequestOutput(), send));
  EXPECT_EQ(sent, 1);
}

class StreamTestService final : public proto::XllmAPIService {
 public:
  void ChatCompletionsHttp(google::protobuf::RpcController* controller,
                           const proto::HttpRequest*,
                           proto::HttpResponse*,
                           google::protobuf::Closure* done) override {
    auto* ctrl = static_cast<brpc::Controller*>(controller);
    const std::string mode = ctrl->request_attachment().to_string();
    proto::ChatRequest request;
    request.set_stream(mode != "named_full");
    request.mutable_stream_options()->set_include_usage(true);
    request.mutable_stream_options()->set_continuous_usage_stats(mode ==
                                                                 "continuous");
    if (mode == "named_full" || mode == "named_stream") {
      request.set_tool_choice(
          R"({"type":"function","function":{"name":"weather"}})");
    } else if (mode == "required_stream") {
      request.set_tool_choice("required");
    }
    proto::ChatResponse response;
    StreamCall<proto::ChatRequest, proto::ChatResponse> call(
        ctrl,
        done,
        &request,
        &response,
        /*use_arena=*/true,
        /*is_http_request=*/true);
    if (mode == "invalid" || mode == "missing" || mode == "limited") {
      call.finish_with_error(mode == "invalid" ? StatusCode::INVALID_ARGUMENT
                             : mode == "missing"
                                 ? StatusCode::NOT_FOUND
                                 : StatusCode::RESOURCE_EXHAUSTED,
                             "failure",
                             mode == "missing" ? "model" : "");
      return;
    }
    if (mode == "named_full") {
      response.set_object("chat.completion");
      response.add_choices()->set_finish_reason("tool_calls");
      call.write_and_finish(response);
      return;
    }
    proto::Usage usage;
    usage.set_prompt_tokens(2);
    usage.set_completion_tokens(1);
    usage.set_total_tokens(3);
    call.set_stream_usage(usage);
    response.set_object("chat.completion.chunk");
    auto* choice = response.add_choices();
    choice->mutable_delta()->set_content("hi");
    if (mode == "named_stream" || mode == "required_stream") {
      choice->set_finish_reason("tool_calls");
    }
    call.write(response);
    if (mode == "stream_error") {
      call.finish_with_error(StatusCode::UNKNOWN, "generation failed");
      return;
    }
    response.clear_choices();
    response.mutable_usage()->set_total_tokens(3);
    call.write(response);
    call.finish();
    call.finish();
  }
};

class OpenAICallTest : public testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(server_.AddService(&service_,
                                 brpc::SERVER_DOESNT_OWN_SERVICE,
                                 "/test => ChatCompletionsHttp"),
              0);
    ASSERT_EQ(server_.Start(/*port=*/0, /*options=*/nullptr), 0);
    brpc::ChannelOptions options;
    options.protocol = brpc::PROTOCOL_HTTP;
    options.timeout_ms = 5000;
    options.max_retry = 0;
    ASSERT_EQ(channel_.Init(server_.listen_address(), &options), 0);
  }
  void TearDown() override {
    server_.Stop(/*timeout_ms=*/0);
    server_.Join();
  }
  void request(const std::string& mode, brpc::Controller& controller) {
    controller.http_request().uri() = "/test";
    controller.http_request().set_method(brpc::HTTP_METHOD_POST);
    controller.request_attachment().append(mode);
    channel_.CallMethod(nullptr, &controller, nullptr, nullptr, nullptr);
  }
  StreamTestService service_;
  brpc::Server server_;
  brpc::Channel channel_;
};

TEST_F(OpenAICallTest, PreStreamErrorsUseJsonAndCorrectHttpStatus) {
  for (const auto& [mode, code] :
       {std::pair{"invalid", 400}, {"missing", 404}, {"limited", 429}}) {
    brpc::Controller controller;
    request(mode, controller);
    EXPECT_EQ(controller.http_response().status_code(), code);
    EXPECT_EQ(controller.http_response().content_type(), "application/json");
    const auto json =
        nlohmann::json::parse(controller.response_attachment().to_string());
    EXPECT_EQ(json["error"]["code"], code);
    EXPECT_EQ(json["error"]["param"],
              std::string(mode) == "missing" ? nlohmann::json("model")
                                             : nlohmann::json(nullptr));
  }
}

TEST_F(OpenAICallTest, NamedToolChoiceIsAppliedToHttpAndSse) {
  for (const std::string mode :
       {"named_full", "named_stream", "required_stream"}) {
    brpc::Controller controller;
    request(mode, controller);
    ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
    const std::string body = controller.response_attachment().to_string();
    const auto json = nlohmann::json::parse(
        mode == "named_full" ? body : body.substr(6, body.find("\n\n") - 6));
    EXPECT_EQ(json["choices"][0]["finish_reason"],
              mode == "required_stream" ? "tool_calls" : "stop");
  }
}

TEST_F(OpenAICallTest, StreamsEndWithExactlyOneDoneAndUsageChoicesArray) {
  brpc::Controller controller;
  request("stream", controller);
  ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
  const std::string body = controller.response_attachment().to_string();
  const auto first =
      nlohmann::json::parse(body.substr(6, body.find("\n\n") - 6));
  EXPECT_FALSE(first.contains("usage"));
  EXPECT_NE(body.find("\"choices\":[]"), std::string::npos);
  const size_t done = body.find("data: [DONE]\n\n");
  ASSERT_NE(done, std::string::npos);
  EXPECT_EQ(body.find("[DONE]", done + 12), std::string::npos);
}

TEST_F(OpenAICallTest, ContinuousUsageIncludesCountsOnContentChunks) {
  brpc::Controller controller;
  request("continuous", controller);
  ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
  const std::string body = controller.response_attachment().to_string();
  const auto first =
      nlohmann::json::parse(body.substr(6, body.find("\n\n") - 6));
  EXPECT_EQ(first["usage"]["prompt_tokens"], 2);
  EXPECT_EQ(first["usage"]["completion_tokens"], 1);
  EXPECT_EQ(first["choices"].size(), 1);
}

TEST_F(OpenAICallTest, GenerationErrorsAreSseJsonAndCloseStream) {
  brpc::Controller controller;
  request("stream_error", controller);
  ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
  const std::string body = controller.response_attachment().to_string();
  EXPECT_NE(body.find("data: {\"error\":"), std::string::npos);
  EXPECT_NE(body.find("generation failed"), std::string::npos);
  EXPECT_TRUE(body.ends_with("data: [DONE]\n\n"));
}

}  // namespace
}  // namespace xllm::api_service
