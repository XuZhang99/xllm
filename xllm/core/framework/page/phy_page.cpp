/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "phy_page.h"

namespace xllm {
PhyPage::PhyPage(int64_t granularity_size, torch::Device device)
    : device_(device) {
  int device_id = device_.index();

#if defined(USE_NPU)
  aclrtPhysicalMemProp prop = {};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_HBM_MEM_HUGE;  // 2MB
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  prop.reserve = 0;

  status_ = aclrtMallocPhysical(&phy_handle_, granularity_size, &prop, 0);
  CHECK_EQ(status_, VmmSuccess) << "Failed to allocate physical memory";
#endif
}

PhyPage::~PhyPage() {
  if (status_ == VmmSuccess) {
#if defined(USE_NPU)
    status_ = aclrtFreePhysical(phy_handle_);
    CHECK_EQ(status_, VmmSuccess) << "Failed to free physical memory";
#endif
  }
}
}  // namespace xllm