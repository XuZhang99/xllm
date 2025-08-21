#pragma once
#include <torch/torch.h>

#include "acl/acl.h"

namespace xllm {
class PhyPage {
 public:
  PhyPage(int64_t granularity_size, torch::Device device);

  ~PhyPage();

  bool is_valid() const { return status_ == ACL_SUCCESS && phy_handle_ != 0; }

  const torch::Device& device() const { return device_; }

  aclrtDrvMemHandle get_phy_handle() const { return phy_handle_; }

 private:
  torch::Device device_;
  aclrtDrvMemHandle phy_handle_;
  aclError status_ = ACL_SUCCESS;
};
}  // namespace xllm