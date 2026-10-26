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

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/macros.h"

namespace xllm {
namespace {

struct NestedEmbeddingArgs final {
  REFLECT_PROPERTIES(NestedEmbeddingArgs);

  PROPERTY(int64_t, vocab_size) = -1;
  PROPERTY(bool, tie_word_embeddings) = false;
};

struct NestedModelArgs final {
  REFLECT_PROPERTIES(NestedModelArgs);

  PROPERTY(NestedEmbeddingArgs, token_embedding_args);
};

struct ReflectionOwnerArgs final {
  REFLECT_PROPERTIES(ReflectionOwnerArgs);

  PROPERTY(NestedModelArgs, model_args);
  PROPERTY(int64_t, owner_width) = 0;
};

class RecordingPropertyVisitor final : public PropertyVisitor {
 public:
  void visit(const std::string& name, bool value) override {
    record_visit(name);
    bool_values_.emplace(name, value);
  }
  void visit(const std::string& name, int32_t) override { record_visit(name); }
  void visit(const std::string& name, int64_t value) override {
    record_visit(name);
    int64_values_.emplace(name, value);
  }
  void visit(const std::string& name, float) override { record_visit(name); }
  void visit(const std::string& name, double) override { record_visit(name); }
  void visit(const std::string& name, const std::string&) override {
    record_visit(name);
  }
  void visit(const std::string& name, const std::vector<int32_t>&) override {
    record_visit(name);
  }
  void visit(const std::string& name, const std::vector<int64_t>&) override {
    record_visit(name);
  }
  void visit(const std::string& name, const std::vector<float>&) override {
    record_visit(name);
  }
  void visit(const std::string& name, const std::vector<double>&) override {
    record_visit(name);
  }
  void visit(const std::string& name, const std::vector<bool>&) override {
    record_visit(name);
  }
  void visit(const std::string& name,
             const std::vector<std::string>&) override {
    record_visit(name);
  }
  void visit(const std::string& name,
             const std::unordered_set<int32_t>&) override {
    record_visit(name);
  }
  void visit_absent(const std::string& name) override { record_visit(name); }

  std::unordered_map<std::string, int32_t> visit_counts_;

  std::unordered_map<std::string, bool> bool_values_;
  std::unordered_map<std::string, int64_t> int64_values_;

 private:
  void record_visit(const std::string& name) { ++visit_counts_[name]; }
};

TEST(PropertyReflectTest, RecursivelyFlattensMultipleLevelsBesideFlatFields) {
  ReflectionOwnerArgs args;
  args.model_args().token_embedding_args().vocab_size(32000);
  args.model_args().token_embedding_args().tie_word_embeddings(true);
  args.owner_width(4096);

  RecordingPropertyVisitor visitor;
  visit_properties(args, visitor);

  EXPECT_EQ(visitor.int64_values_.at("vocab_size"), 32000);
  EXPECT_TRUE(visitor.bool_values_.at("tie_word_embeddings"));
  EXPECT_EQ(visitor.int64_values_.at("owner_width"), 4096);
  EXPECT_EQ(visitor.visit_counts_.at("vocab_size"), 1);
  EXPECT_EQ(visitor.visit_counts_.at("tie_word_embeddings"), 1);
  EXPECT_EQ(visitor.visit_counts_.at("owner_width"), 1);
  EXPECT_EQ(visitor.int64_values_.count("model_args.vocab_size"), 0);
  EXPECT_EQ(
      visitor.int64_values_.count("model_args.token_embedding_args.vocab_size"),
      0);
  EXPECT_EQ(visitor.bool_values_.count("model_args.tie_word_embeddings"), 0);
  EXPECT_EQ(visitor.bool_values_.count(
                "model_args.token_embedding_args.tie_word_embeddings"),
            0);
}

}  // namespace
}  // namespace xllm
