#include "phy_page.h"

namespace xllm {
PhyPage::PhyPage(int64_t granularity_size, torch::Device device)
    : device_(device) {
  int device_id = device_.index();

  aclrtPhysicalMemProp prop = {};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_HBM_MEM_HUGE;  // 2MB
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  prop.reserve = 0;

  status_ = aclrtMallocPhysical(&phy_handle_, granularity_size, &prop, 0);
  CHECK_EQ(status_, ACL_SUCCESS) << "Failed to allocate physical memory";
}

PhyPage::~PhyPage() {
  if (status_ == ACL_SUCCESS) {
    status_ = aclrtFreePhysical(phy_handle_);
    CHECK_EQ(status_, ACL_SUCCESS) << "Failed to free physical memory";
  }
}
}  // namespace xllm