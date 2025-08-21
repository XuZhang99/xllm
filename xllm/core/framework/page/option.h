#pragma once

#include <torch/torch.h>

#include <vector>

#include "common/macros.h"

namespace xllm {
namespace page {
struct Options {
  // devices for page manager pool
  PROPERTY(std::vector<torch::Device>, devices);

  // devices for page manager
  PROPERTY(torch::Device, device);

  // granularity size for one page in bytes
  PROPERTY(int64_t, granularity_size) = 2 * 1024 * 1024;  // 2MB

  // num of layers
  PROPERTY(int64_t, num_layers) = 0;

  // total pages for page manager
  PROPERTY(int64_t, total_pages) = 0;

  // key or value cache size in bytes per token
  PROPERTY(int64_t, cache_size_per_token) = 0;
};
}  // namespace page
}  // namespace xllm