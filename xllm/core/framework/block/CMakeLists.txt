include(cc_library)
include(cc_test)

cc_library(
  NAME 
    block
  HDRS
    block.h
    block_manager.h
    block_manager_pool.h
    block_manager_impl.h
    concurrent_block_manager_impl.h
  SRCS 
    block.cpp
    block_manager_pool.cpp
    concurrent_block_manager_impl.cpp
    block_manager_impl.cpp
  DEPS
    :request
    :common
    glog::glog
    Boost::serialization
    SMHasherSupport
    torch
)

cc_test(
  NAME
    block_test
  SRCS
    block_manager_test.cpp
  DEPS
    :block
    GTest::gtest_main
)

