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

#include "core/framework/speculative/draft_config.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace xllm {

namespace {

TEST(DraftConfigTest, BoundaryToPostLayerIdsShiftsAndRejectsEmbedding) {
  // Speculators boundary indices (0=embedding, v=output of layer v-1) shift
  // to 0-based post-layer capture indices.
  EXPECT_EQ(boundary_to_post_layer_ids({2, 32, 61}),
            (std::vector<int32_t>{1, 31, 60}));
  EXPECT_TRUE(boundary_to_post_layer_ids({}).empty());

  // Boundary 0 (the embedding output) is legal in speculators but xLLM only
  // captures decoder-layer outputs; it must be rejected loudly instead of
  // shifted to -1, where it would never match and silently feed the draft
  // uninitialized capture slots.
  EXPECT_DEATH(boundary_to_post_layer_ids({0, 32, 61}), "Embedding capture");
}

}  // namespace

}  // namespace xllm
