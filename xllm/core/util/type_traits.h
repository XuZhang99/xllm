#pragma once
#include <optional>

#include "framework/page/multi_layer_xtensor.h"

namespace xllm {

template <typename value_type>
struct remove_optional {
  using type = value_type;
};

// specialization for optional
template <typename value_type>
struct remove_optional<std::optional<value_type>> {
  using type = value_type;
};

/// alias template for remove_optional
template <typename value_type>
using remove_optional_t = typename remove_optional<value_type>::type;

using MultiLayerXTensorPair = std::pair<std::unique_ptr<MultiLayerXTensor>,
                                        std::unique_ptr<MultiLayerXTensor>>;
}  // namespace xllm