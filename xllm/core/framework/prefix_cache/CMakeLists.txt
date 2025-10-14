include(cc_binary)
include(cc_library)
include(cc_test)

cc_library(
  NAME 
    prefix_cache
  HDRS
    prefix_cache.h
    prefix_cache_with_upload.h
    prefix_cache_factory.h
  SRCS 
    prefix_cache.cpp
    prefix_cache_with_upload.cpp
    prefix_cache_factory.cpp
  DEPS
    $<$<BOOL:${USE_NPU}>:graph>
    :request
    :common
    SMHasherSupport
)


cc_test(
  NAME
    prefix_test
  SRCS
    prefix_cache_test.cpp
  DEPS
    :prefix_cache
    GTest::gtest_main
)


cc_binary(
  NAME
    prefix_cache_benchmark
  SRCS
    prefix_cache_benchmark.cpp
  DEPS
    :kv_cache
    :prefix_cache
    :block
    benchmark::benchmark
    benchmark::benchmark_main
)
