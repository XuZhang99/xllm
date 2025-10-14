include(cc_library)
include(cc_test)

cc_library(
  NAME 
    eplb
  HDRS
    eplb_executor.h
    eplb_manager.h
    eplb_policy.h
    expert_weight_buffer_shm.h
    shared_memory_manager.h
    expert_buffer_manager.h
  SRCS 
    eplb_executor.cpp
    eplb_manager.cpp
    eplb_policy.cpp
    expert_weight_buffer_shm.cpp
    shared_memory_manager.cpp
    expert_buffer_manager.cpp
  DEPS
    :common
    :platform
)

cc_test(
  NAME
    eplb_policy_test
  SRCS
    eplb_policy_test.cpp
  DEPS
    :eplb
    fmt::fmt
    GTest::gtest_main
)

